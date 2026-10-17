//! Minimal Metal runtime for PrismML Ternary Bonsai 2 (2B and 4B, sizes from config.json): one architecture,
//! greedy decoding, text plus images, optional LoRA adapter.
//! Single turn by design: optional system prompt (prefilled once and cached) plus user input, optional image.

pub mod gpu;
pub mod lora;
pub mod model;
pub mod tokenizer;
pub mod vision;
pub mod weights;
pub(crate) mod ext;

use gpu::Gpu;
use model::{State, Text, TMAX};
use std::path::Path;
use std::time::Instant;
use tokenizer::{Tokenizer, Utf8Stream};
use weights::Weights;

pub const EOS: [u32; 2] = [248046, 248044];
pub const IMAGE_PAD: u32 = 248056;

/// Qwen3.5 chat format pieces (from chat_template.jinja, thinking disabled). Kept as data so other turn
/// shapes (tools) can be built without code changes.
pub struct Template {
    pub system: (&'static str, &'static str),
    pub user: (&'static str, &'static str),
    pub assistant: &'static str,
    pub image: &'static str,
}

pub const QWEN35: Template = Template {
    system: ("<|im_start|>system\n", "<|im_end|>\n"),
    user: ("<|im_start|>user\n", "<|im_end|>\n"),
    assistant: "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    image: "<|vision_start|><|image_pad|><|vision_end|>",
};

/// Prompt-lookup speculation: draft the tokens that followed the latest earlier occurrence of the last
/// n-gram (n from `ngram_max` down to `ngram_min`) in the request's prompt plus output.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub ngram_max: usize,
    pub ngram_min: usize,
    pub draft_max: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            ngram_max: 3,
            ngram_min: 1,
            draft_max: 7,
        }
    }
}

/// First earlier occurrence of the last n tokens of `seq` (n = ngram_max down to ngram_min) inside
/// `seq[..user_len]` (the request's user turn; the cached system prompt is never searched), per results/spec.md.
fn lookup(seq: &[u32], user_len: usize, sp: &Spec, limit: usize) -> Vec<u32> {
    let k = sp.draft_max.min(limit).min(model::VMAX - 1);
    for n in (sp.ngram_min..=sp.ngram_max).rev() {
        if seq.len() <= n {
            continue;
        }
        let pat = &seq[seq.len() - n..];
        let last = user_len.min(seq.len() - n);
        for i in 0..last.saturating_sub(n - 1) {
            if &seq[i..i + n] == pat {
                let end = (i + n + k).min(user_len);
                if end > i + n {
                    return seq[i + n..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Encodes a deferred `accept` (see `Model::defer_accept`) into `cmd`.
fn encode_pending(pending: &mut Option<(usize, usize)>, text: &mut Text, wb: &gpu::Buffer, cmd: &mut gpu::Cmd) {
    if let Some((t, n)) = pending.take() {
        text.encode_accept(cmd, wb, t, n);
    }
}

/// Verify cost in decode steps for T = 1..=8 rows (the last token plus T - 1 drafts), accept included
/// (`bonsai --verify-bench`, M2 Pro, results/rust.md "Optimization 2"). T = 2, 3 use qmvr, T >= 4 the MMA kernel.
const VERIFY_COST_2B: [f64; 8] = [1.0, 1.57, 2.1, 2.6, 2.6, 2.6, 2.65, 2.65];
const VERIFY_COST_4B: [f64; 8] = [1.0, 1.6, 2.15, 2.8, 2.85, 2.9, 2.95, 3.0];
/// Expected gain (tokens per step relative to plain decoding) needed to keep speculating, and the higher bar
/// for switching back from plain decoding (hysteresis). The margin covers what the cost table leaves out
/// (a host sync per round where plain decoding is pipelined, estimate noise); 1.0 lost 5% on the 4B w120 case.
const KEEP_GAIN: f64 = 1.05;
const ENTER_GAIN: f64 = 1.15;
/// Backstop: after this many verify rounds, a request whose rounds returned fewer than KEEP_GAIN tokens per
/// step of table cost decodes plainly to the end.
const GIVE_UP_ROUNDS: usize = 6;

/// Adaptive speculation policy. Drafts are accepted left to right, so with a per-draft acceptance p a round
/// of k drafts yields E(k) = (1 - p^(k+1)) / (1 - p) tokens for cost[k] steps. Each round picks the k with
/// the best E(k) / cost[k]; k = 0 (plain decoding, 1 token per step) when none clears the margin. p comes from
/// accepted and rejected drafts of recent rounds (decayed per round) plus a prior from results/spec.md.
struct Policy {
    cost: &'static [f64; 8],
    kmax: usize,
    acc: f64,
    rej: f64,
}

impl Policy {
    /// Prior: p = 0.85 worth 4 drafts (spec.md: 80-88% of drafts accepted at k = 2..4 on cleanup prompts).
    const PRIOR_ACC: f64 = 3.4;
    const PRIOR_REJ: f64 = 0.6;
    /// Per-round decay of the counts: the estimate follows the last ~7 rounds (output that stops copying the
    /// input, like the run-on tail after a cleanup, drops it within a few rounds).
    const DECAY: f64 = 0.85;

    fn new(cost: &'static [f64; 8], kmax: usize) -> Policy {
        Policy { cost, kmax: kmax.min(7), acc: 0.0, rej: 0.0 }
    }

    fn p(&self) -> f64 {
        (self.acc + Self::PRIOR_ACC) / (self.acc + self.rej + Self::PRIOR_ACC + Self::PRIOR_REJ)
    }

    /// Best draft length up to `limit` and its expected tokens per step; (0, 1) if none beats `min_gain`.
    fn choose(&self, limit: usize, min_gain: f64) -> (usize, f64) {
        let p = self.p();
        let mut best = (0, 1.0);
        for k in 1..=self.kmax.min(limit) {
            let g = (1.0 - p.powi(k as i32 + 1)) / (1.0 - p) / self.cost[k];
            if g > best.1 && g > min_gain {
                best = (k, g);
            }
        }
        best
    }

    /// A round (real or shadow) drafted k tokens and m were accepted.
    fn update(&mut self, k: usize, m: usize) {
        self.acc = self.acc * Self::DECAY + m as f64;
        self.rej = self.rej * Self::DECAY + (m < k) as u32 as f64;
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub gen_tokens: usize,
    pub vision_s: f64,
    pub prefill_s: f64,
    pub ttft_s: f64,
    pub decode_s: f64,
    pub verify_calls: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Plain-to-speculation switches of the adaptive policy.
    pub switches: usize,
}

pub struct Model {
    pub gpu: Gpu,
    pub w: Weights,
    pub text: Text,
    pub tok: Tokenizer,
    pub template: Template,
    vision: Option<vision::Vision>,
    prefix: Option<(Vec<u32>, State, usize, i32)>,
    dir: std::path::PathBuf,
    /// Prompt-lookup speculation for generate() (on by default; draft length chosen per round by `Policy`, up
    /// to `draft_max`; output is token-identical to plain greedy); None decodes one token per step, pipelined.
    pub spec: Option<Spec>,
    /// Deferred accept (t, n) of the last verify.
    pending: Option<(usize, usize)>,
    /// Benchmarks only: keep decoding past EOS.
    pub ignore_eos: bool,
}

impl Model {
    pub fn load(dir: &Path, ctx: usize) -> Result<Model, String> {
        let t = Instant::now();
        let gpu = Gpu::new()?;
        let t_gpu = t.elapsed();
        let w = Weights::open(&dir.join("model.safetensors"), &gpu)?;
        let t_w = t.elapsed();
        let cfg = model::Config::load(&dir.join("config.json"))?;
        cfg.check()?;
        let text = Text::new(&gpu, &w, &cfg, ctx);
        let t_text = t.elapsed();
        let tok = Tokenizer::load(&dir.join("tokenizer.json"), &dir.join("tokenizer.bonsai"))?;
        if std::env::var_os("BONSAI_LOADTIME").is_some() {
            eprintln!(
                "gpu+kernels {:?} weights {:?} text {:?} tokenizer {:?}",
                t_gpu,
                t_w - t_gpu,
                t_text - t_w,
                t.elapsed() - t_text
            );
        }
        Ok(Model {
            gpu,
            w,
            text,
            tok,
            template: QWEN35,
            vision: None,
            prefix: None,
            dir: dir.to_path_buf(),
            spec: Some(Spec::default()),
            pending: None,
            ignore_eos: false,
        })
    }

    /// Applies a LoRA adapter (safetensors, see lora.rs) with B scaled by `scale`; None or scale 0 runs the base.
    /// Drops the cached system prompt state.
    pub fn set_lora(&mut self, path: Option<&Path>, scale: f32) -> Result<(), String> {
        let l = match path {
            Some(p) if scale != 0.0 => Some(lora::load(p, scale, &self.text, &self.gpu)?),
            _ => None,
        };
        self.text.set_lora(&self.gpu, l);
        self.prefix = None;
        Ok(())
    }

    /// Runs `ids` (with optional spliced embeddings) through the model; returns argmax after the last row.
    /// `rows` are host embeddings for some rows: (row index, hidden-size floats).
    pub fn prefill(&mut self, ids: &[u32], pos: &[[i32; 3]], rows: &[(usize, &[f32])]) -> u32 {
        self.flush_accept();
        let mut next = 0;
        let mut i = 0;
        let d = self.text.cfg.hidden;
        while i < ids.len() {
            let t = (ids.len() - i).min(TMAX);
            self.text.write_ids(&ids[i..i + t]);
            self.text.write_pos(&pos[i..i + t]);
            let wb = &self.w.buf;
            let mut cmd = self.gpu.begin();
            self.text.encode_embed(&mut cmd, wb, t);
            let spliced: Vec<_> = rows.iter().filter(|(r, _)| *r >= i && *r < i + t).collect();
            if !spliced.is_empty() {
                cmd.finish();
                let h = self.text.hidden_ptr();
                for (r, v) in spliced {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            v.as_ptr(),
                            h.add((r - i) * d),
                            d,
                        )
                    };
                }
                cmd = self.gpu.begin();
            }
            self.text.encode_layers(&mut cmd, wb, t, false);
            let last = i + t == ids.len();
            if last {
                self.text.encode_head(&mut cmd, wb, t - 1, 1);
            }
            cmd.finish();
            self.text.state.len += t;
            i += t;
            if last {
                next = self.text.argmax(1)[0];
            }
        }
        if let Some(p) = pos.last() {
            self.text.state.pos = p.iter().copied().max().unwrap() + 1;
        }
        next
    }

    /// One decode step: feed `id`, return the greedy next token.
    pub fn step(&mut self, id: u32) -> u32 {
        let mut cmd = self.gpu.begin();
        encode_pending(&mut self.pending, &mut self.text, &self.w.buf, &mut cmd);
        let p = self.text.state.pos;
        self.text.write_ids(&[id]);
        self.text.write_pos(&[[p, p, p]]);
        let wb = &self.w.buf;
        self.text.encode_embed(&mut cmd, wb, 1);
        self.text.encode_layers(&mut cmd, wb, 1, false);
        self.text.encode_head(&mut cmd, wb, 0, 1);
        cmd.finish();
        self.text.state.len += 1;
        self.text.state.pos += 1;
        self.text.argmax(1)[0]
    }

    /// Feeds `ids` (last emitted token plus drafts) in one forward and returns the argmax at every row.
    /// State is left untouched until `accept`.
    pub fn verify(&mut self, ids: &[u32]) -> Vec<u32> {
        let mut cmd = self.gpu.begin();
        encode_pending(&mut self.pending, &mut self.text, &self.w.buf, &mut cmd);
        let t = ids.len();
        let p = self.text.state.pos;
        let pos: Vec<[i32; 3]> = (0..t as i32).map(|k| [p + k; 3]).collect();
        self.text.write_ids(ids);
        self.text.write_pos(&pos);
        let wb = &self.w.buf;
        self.text.encode_embed(&mut cmd, wb, t);
        self.text.encode_layers(&mut cmd, wb, t, true);
        self.text.encode_head(&mut cmd, wb, 0, t);
        cmd.finish();
        self.text.argmax(t)
    }

    /// Commits the first `n` rows of the last `verify` call of `t` rows.
    pub fn accept(&mut self, t: usize, n: usize) {
        self.defer_accept(t, n);
        self.flush_accept();
    }

    /// Like `accept`, but the commit is encoded at the start of the next `verify` or `step` command buffer,
    /// which saves a submit and wait per speculation round. `flush_accept` runs it on its own.
    pub fn defer_accept(&mut self, t: usize, n: usize) {
        assert!(self.pending.is_none());
        self.pending = Some((t, n));
    }

    pub fn flush_accept(&mut self) {
        if self.pending.is_some() {
            let mut cmd = self.gpu.begin();
            encode_pending(&mut self.pending, &mut self.text, &self.w.buf, &mut cmd);
            cmd.finish();
        }
    }

    pub fn reset(&mut self) {
        self.pending = None;
        self.text.reset();
    }

    fn text_positions(start: i32, n: usize) -> Vec<[i32; 3]> {
        (0..n as i32).map(|k| [start + k; 3]).collect()
    }

    /// Restores the cached state after `prefix` (token ids), or prefills and caches it. Returns the
    /// number of tokens served from the cache.
    pub fn start_prefix(&mut self, prefix: &[u32]) -> usize {
        self.pending = None;
        if let Some((k, snap, len, pos)) = &self.prefix {
            if k.as_slice() == prefix {
                let mut cmd = self.gpu.begin();
                self.text
                    .encode_copy_state(&mut cmd, snap, &self.text.state);
                cmd.finish();
                self.text.state.len = *len;
                self.text.state.pos = *pos;
                return *len;
            }
        }
        self.reset();
        if !prefix.is_empty() {
            let pos = Self::text_positions(0, prefix.len());
            self.prefill(prefix, &pos, &[]);
        }
        let mut snap = State::new(&self.gpu, &self.text.cfg, self.text.ctx, self.text.n_gdn, self.text.n_attn);
        snap.len = self.text.state.len;
        let mut cmd = self.gpu.begin();
        self.text
            .encode_copy_state(&mut cmd, &self.text.state, &snap);
        cmd.finish();
        self.prefix = Some((
            prefix.to_vec(),
            snap,
            self.text.state.len,
            self.text.state.pos,
        ));
        0
    }

    fn start(&mut self, system: Option<&str>) -> usize {
        let ids = match system {
            Some(s) => {
                let t = &self.template;
                self.tok
                    .encode(&format!("{}{}{}", t.system.0, s.trim(), t.system.1))
            }
            None => Vec::new(),
        };
        self.start_prefix(&ids)
    }

    pub fn clear_prefix_cache(&mut self) {
        self.prefix = None;
    }

    /// System prompt + user input (+ image) -> greedy output, streamed through `emit`.
    pub fn generate(
        &mut self,
        system: Option<&str>,
        user: &str,
        image: Option<&Path>,
        max_tokens: usize,
        emit: impl FnMut(&str),
    ) -> Result<(Vec<u32>, Stats), String> {
        let t0 = Instant::now();
        let mut st = Stats::default();
        // Vision first: it does not depend on the text state.
        let img = match image {
            Some(p) => {
                if self.vision.is_none() {
                    self.vision = Some(vision::Vision::new(&self.gpu, &self.w));
                }
                let v = self.vision.as_ref().unwrap();
                let out = v.run(&self.gpu, &self.w.buf, p)?;
                st.vision_s = t0.elapsed().as_secs_f64();
                Some(out)
            }
            None => None,
        };
        st.cached_tokens = self.start(system);
        let t = &self.template;
        let mut text = String::from(t.user.0);
        if img.is_some() {
            text.push_str(t.image);
        }
        text.push_str(user.trim());
        text.push_str(t.user.1);
        text.push_str(t.assistant);
        let raw = self.tok.encode(&text);
        self.generate_after_prefix(raw, img.as_ref(), max_tokens, t0, st, emit)
    }

    /// Prefills `raw` (user turn token ids, image pads expanded from `img`) on top of the current state
    /// and decodes greedily.
    pub fn generate_after_prefix(
        &mut self,
        raw: Vec<u32>,
        img: Option<&vision::ImageOut>,
        max_tokens: usize,
        t0: Instant,
        mut st: Stats,
        mut emit: impl FnMut(&str),
    ) -> Result<(Vec<u32>, Stats), String> {
        let p0 = self.text.state.pos;
        let mut ids = Vec::with_capacity(raw.len());
        let mut pos = Vec::with_capacity(raw.len());
        let mut rows: Vec<(usize, &[f32])> = Vec::new();
        let mut p = p0;
        let d = self.text.cfg.hidden;
        for &id in &raw {
            if id == IMAGE_PAD {
                if let Some(im) = img {
                    let (gh, gw) = (im.grid.1 / 2, im.grid.2 / 2);
                    for r in 0..gh {
                        for c in 0..gw {
                            rows.push((
                                ids.len(),
                                &im.embeds[(r * gw + c) * d..][..d],
                            ));
                            ids.push(IMAGE_PAD);
                            pos.push([p, p + r as i32, p + c as i32]);
                        }
                    }
                    p += gh.max(gw) as i32;
                    continue;
                }
            }
            ids.push(id);
            pos.push([p; 3]);
            p += 1;
        }
        st.prompt_tokens = ids.len();
        let tp = Instant::now();
        let next = self.prefill(&ids, &pos, &rows);
        st.prefill_s = tp.elapsed().as_secs_f64();
        st.ttft_s = t0.elapsed().as_secs_f64();
        let td = Instant::now();
        let mut out = Vec::new();
        let mut stream = Utf8Stream::default();
        if max_tokens > 0 && !EOS.contains(&next) {
            out.push(next);
            emit(&stream.push(self.tok.token_bytes(next)));
            match self.spec {
                Some(sp) => self.decode_spec(sp, &raw, &mut out, max_tokens, &mut st, &mut stream, &mut emit),
                None => {
                    self.decode_plain(next, &mut out, max_tokens, &mut stream, &mut emit, |_| false);
                }
            }
        }
        st.decode_s = td.elapsed().as_secs_f64();
        st.gen_tokens = out.len();
        Ok((out, st))
    }

    /// Prompt-lookup speculation with the adaptive policy (`Policy`): each round drafts the k that maximizes
    /// expected tokens per unit of verify cost, or decodes plainly (pipelined) while speculation does not pay,
    /// scoring shadow rounds on the real output to decide when to switch back. `out` holds the first token.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec(
        &mut self,
        sp: Spec,
        user: &[u32],
        out: &mut Vec<u32>,
        max_tokens: usize,
        st: &mut Stats,
        stream: &mut Utf8Stream,
        emit: &mut impl FnMut(&str),
    ) {
        let cost = if self.text.cfg.hidden <= 2048 { &VERIFY_COST_2B } else { &VERIFY_COST_4B };
        let mut pol = Policy::new(cost, sp.draft_max);
        let mut seq = user.to_vec();
        seq.extend_from_slice(out);
        let ulen = user.len();
        // realized tokens and table cost of the verify rounds so far
        let (mut got, mut spent, mut rounds) = (0.0, 0.0, 0);
        while out.len() < max_tokens {
            if ext::cancelled() {
                return;
            }
            let last = *seq.last().unwrap();
            let left = max_tokens - out.len();
            if rounds >= GIVE_UP_ROUNDS && got < KEEP_GAIN * spent {
                // Acceptance comes in bursts the estimate keeps chasing: plain for the rest of the request.
                self.decode_plain(last, out, max_tokens, stream, emit, |_| false);
                return;
            }
            let k = pol.choose(left - 1, KEEP_GAIN).0;
            if k == 0 {
                // Plain decoding; shadow rounds (what lookup would have drafted vs what came out) update the
                // estimate, and speculation resumes once it clears the entry margin.
                let (mut next_start, mut sh_pos, mut sh): (usize, usize, Vec<u32>) = (seq.len(), 0, Vec::new());
                let stop = |id: u32| {
                    seq.push(id);
                    let j = seq.len() - 1;
                    if !sh.is_empty() && j >= sh_pos {
                        let i = j - sh_pos;
                        if seq[j] != sh[i] {
                            pol.update(sh.len(), i);
                            sh.clear();
                            next_start = j + 1;
                        } else if i + 1 == sh.len() {
                            pol.update(sh.len(), sh.len());
                            sh.clear();
                            next_start = j + 2;
                        }
                    }
                    if sh.is_empty() && seq.len() >= next_start {
                        sh = lookup(&seq, ulen, &sp, usize::MAX);
                        sh_pos = seq.len();
                        next_start = seq.len() + 1;
                    }
                    pol.choose(sp.draft_max, ENTER_GAIN).0 > 0
                };
                match self.decode_plain(last, out, max_tokens, stream, emit, stop) {
                    Some(id) => {
                        seq.push(id);
                        st.switches += 1;
                        continue;
                    }
                    None => return,
                }
            }
            let draft = lookup(&seq, ulen, &Spec { draft_max: k, ..sp }, left - 1);
            let new = if draft.is_empty() {
                vec![self.step(last)]
            } else {
                let mut ids = vec![last];
                ids.extend_from_slice(&draft);
                let pred = self.verify(&ids);
                let m = draft.iter().zip(&pred).take_while(|(d, p)| d == p).count();
                self.defer_accept(ids.len(), m + 1);
                pol.update(draft.len(), m);
                got += (m + 1) as f64;
                spent += cost[draft.len()];
                rounds += 1;
                st.verify_calls += 1;
                st.drafted += draft.len();
                st.accepted += m;
                let mut v = draft[..m].to_vec();
                v.push(pred[m]);
                v
            };
            for id in new {
                if (EOS.contains(&id) && !self.ignore_eos) || out.len() >= max_tokens {
                    return;
                }
                out.push(id);
                seq.push(id);
                emit(&stream.push(self.tok.token_bytes(id)));
            }
        }
    }

    /// Plain greedy decoding with two steps in flight: step k reads its token from GPU ring slot k % 2 and
    /// writes the next to (k + 1) % 2, so there is no host round trip per token. `stop(id)` is called for every
    /// emitted token; when it returns true the step in flight is drained and its token (emitted, not yet fed)
    /// is returned. None: EOS or the token cap.
    fn decode_plain(
        &mut self,
        feed: u32,
        out: &mut Vec<u32>,
        max_tokens: usize,
        stream: &mut Utf8Stream,
        emit: &mut impl FnMut(&str),
        mut stop: impl FnMut(u32) -> bool,
    ) -> Option<u32> {
        if out.len() >= max_tokens {
            return None;
        }
        self.flush_accept();
        self.text.set_ring(0, feed);
        let mut k = 0;
        let mut cmd = self.gpu.begin();
        self.text.encode_step(&mut cmd, &self.w.buf, k);
        let mut prev = cmd.commit();
        while out.len() < max_tokens {
            if ext::cancelled() {
                prev.wait();
                break;
            }
            k += 1;
            let cur = if out.len() + 1 < max_tokens {
                let mut cmd = self.gpu.begin();
                self.text.encode_step(&mut cmd, &self.w.buf, k);
                Some(cmd.commit())
            } else {
                None
            };
            prev.wait();
            let id = self.text.ring(k);
            if EOS.contains(&id) && !self.ignore_eos {
                if let Some(c) = cur {
                    c.wait();
                }
                break;
            }
            out.push(id);
            emit(&stream.push(self.tok.token_bytes(id)));
            let Some(c) = cur else { break };
            if stop(id) {
                c.wait();
                let id = self.text.ring(k + 1);
                if (EOS.contains(&id) && !self.ignore_eos) || out.len() >= max_tokens {
                    return None;
                }
                out.push(id);
                emit(&stream.push(self.tok.token_bytes(id)));
                return Some(id);
            }
            prev = c;
        }
        None
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
