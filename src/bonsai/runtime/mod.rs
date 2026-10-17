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
    /// Prompt-lookup speculation for generate() (on by default, k = 7: a verify of up to 8 rows costs the same,
    /// output is token-identical to plain greedy); None decodes one token per step, pipelined.
    pub spec: Option<Spec>,
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
        let p = self.text.state.pos;
        self.text.write_ids(&[id]);
        self.text.write_pos(&[[p, p, p]]);
        let wb = &self.w.buf;
        let mut cmd = self.gpu.begin();
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
        let t = ids.len();
        let p = self.text.state.pos;
        let pos: Vec<[i32; 3]> = (0..t as i32).map(|k| [p + k; 3]).collect();
        self.text.write_ids(ids);
        self.text.write_pos(&pos);
        let wb = &self.w.buf;
        let mut cmd = self.gpu.begin();
        self.text.encode_embed(&mut cmd, wb, t);
        self.text.encode_layers(&mut cmd, wb, t, true);
        self.text.encode_head(&mut cmd, wb, 0, t);
        cmd.finish();
        self.text.argmax(t)
    }

    /// Commits the first `n` rows of the last `verify` call of `t` rows.
    pub fn accept(&mut self, t: usize, n: usize) {
        let mut cmd = self.gpu.begin();
        self.text.encode_accept(&mut cmd, &self.w.buf, t, n);
        cmd.finish();
    }

    pub fn reset(&mut self) {
        self.text.reset();
    }

    fn text_positions(start: i32, n: usize) -> Vec<[i32; 3]> {
        (0..n as i32).map(|k| [start + k; 3]).collect()
    }

    /// Restores the cached state after `prefix` (token ids), or prefills and caches it. Returns the
    /// number of tokens served from the cache.
    pub fn start_prefix(&mut self, prefix: &[u32]) -> usize {
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
            // `feed`: last emitted token, not yet run through the model.
            let mut feed = Some(next);
            if let Some(sp) = self.spec {
                feed = self.decode_spec(sp, &raw, next, &mut out, max_tokens, &mut st, &mut stream, &mut emit);
            }
            if let Some(f) = feed {
                self.decode_plain(f, &mut out, max_tokens, &mut stream, &mut emit);
            }
        }
        st.decode_s = td.elapsed().as_secs_f64();
        st.gen_tokens = out.len();
        Ok((out, st))
    }

    /// Prompt-lookup speculation. Returns the unfed last token if it gives up (acceptance under 20% after 3
    /// rounds: output that does not copy the input), so plain pipelined decoding can continue; None when done.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec(
        &mut self,
        sp: Spec,
        user: &[u32],
        first: u32,
        out: &mut Vec<u32>,
        max_tokens: usize,
        st: &mut Stats,
        stream: &mut Utf8Stream,
        emit: &mut impl FnMut(&str),
    ) -> Option<u32> {
        let mut seq = user.to_vec();
        seq.push(first);
        while out.len() < max_tokens {
            if ext::cancelled() {
                return None;
            }
            let last = *seq.last().unwrap();
            if st.verify_calls >= 3 && st.accepted * 5 < st.drafted {
                return Some(last);
            }
            let draft = lookup(&seq, user.len(), &sp, max_tokens - out.len());
            let new = if draft.is_empty() {
                vec![self.step(last)]
            } else {
                let mut ids = vec![last];
                ids.extend_from_slice(&draft);
                let pred = self.verify(&ids);
                let m = draft.iter().zip(&pred).take_while(|(d, p)| d == p).count();
                self.accept(ids.len(), m + 1);
                st.verify_calls += 1;
                st.drafted += draft.len();
                st.accepted += m;
                let mut v = draft[..m].to_vec();
                v.push(pred[m]);
                v
            };
            for id in new {
                if (EOS.contains(&id) && !self.ignore_eos) || out.len() >= max_tokens {
                    return None;
                }
                out.push(id);
                seq.push(id);
                emit(&stream.push(self.tok.token_bytes(id)));
            }
        }
        None
    }

    /// Plain greedy decoding with two steps in flight: step k reads its token from GPU ring slot k % 2 and
    /// writes the next to (k + 1) % 2, so there is no host round trip per token.
    fn decode_plain(
        &mut self,
        feed: u32,
        out: &mut Vec<u32>,
        max_tokens: usize,
        stream: &mut Utf8Stream,
        emit: &mut impl FnMut(&str),
    ) {
        if out.len() >= max_tokens {
            return;
        }
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
            match cur {
                Some(c) => prev = c,
                None => break,
            }
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
