//! Minimal Metal runtime for PrismML Ternary Bonsai 2 (2B and 4B, sizes from config.json): one architecture,
//! greedy decoding, text plus images, optional LoRA adapter.
//! Single turn by design: optional system prompt (prefilled once and cached) plus user input, optional image.

#[cfg(target_os = "macos")]
pub mod gpu;
#[cfg(not(target_os = "macos"))]
#[path = "gpu_vk.rs"]
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

/// Prompt-lookup speculation: draft the source tokens that followed an occurrence of the output's last n words
/// (n from `ngram_max` down to `ngram_min`); see `lookup` and `Model::draft_source`.
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

/// Draft source: the request's source tokens plus a matching key per word token (lowercase, without the
/// leading space, so "Sam" after a cleanup still finds "sam"; punctuation-only tokens get no key and are
/// skipped when matching, so an inserted comma does not break the match).
struct Source {
    ids: Vec<u32>,
    /// (key, index into ids) of the word tokens
    words: Vec<(u64, usize)>,
}

/// Matching key of a token: bit 0 set for word tokens (the only ones matched), bit 1 set when the token
/// contains a line break.
fn token_key(bytes: &[u8]) -> u64 {
    let nl = if bytes.contains(&b'\n') { 2 } else { 0 };
    let b = bytes.strip_prefix(b" ").unwrap_or(bytes);
    if b.iter().all(|c| c.is_ascii_punctuation() || c.is_ascii_whitespace()) {
        return nl;
    }
    let h = b.iter().fold(0xcbf29ce484222325u64, |h, c| (h ^ c.to_ascii_lowercase() as u64).wrapping_mul(0x100000001b3));
    (h & !3) | 1 | nl
}

/// Drafts for the output so far (`seq`, which ends with the output): the source tokens that followed an
/// occurrence of the last n word keys of `seq` (n = ngram_max down to ngram_min). Takes the first occurrence
/// that ends at or after `cursor` (where the previous draft's accepted part ended), else the first one.
/// Returns the draft and the source position it starts at.
fn lookup(src: &Source, seq: &[u32], keys: &[u64], sp: &Spec, limit: usize, cursor: usize) -> (Vec<u32>, usize) {
    let k = sp.draft_max.min(limit).min(model::VMAX - 1);
    if k == 0 {
        return (Vec::new(), cursor);
    }
    let mut pat = Vec::with_capacity(sp.ngram_max);
    for &id in seq.iter().rev().take(4 * sp.ngram_max) {
        if pat.len() == sp.ngram_max {
            break;
        }
        let key = keys.get(id as usize).copied().unwrap_or(0);
        if key & 1 != 0 {
            pat.push(key);
        }
    }
    pat.reverse();
    let w = &src.words;
    for n in (sp.ngram_min..=sp.ngram_max.min(pat.len())).rev() {
        let p = &pat[pat.len() - n..];
        let mut hit = None;
        for i in 0..w.len().saturating_sub(n - 1) {
            if w[i..i + n].iter().zip(p).all(|(a, b)| a.0 == *b) {
                let j = w[i + n - 1].1 + 1;
                // the output moves forward through the source: a match far behind it is a different spot
                if j < src.ids.len() && j + BACK >= cursor && (hit.is_none() || j >= cursor) {
                    hit = Some(j);
                }
                if j >= cursor {
                    break;
                }
            }
        }
        if let Some(j) = hit {
            return (src.ids[j..(j + k).min(src.ids.len())].to_vec(), j);
        }
    }
    (Vec::new(), cursor)
}

/// How far behind the cursor a lookup match may start (self-corrections repeat a few words).
const BACK: usize = 24;

/// Delimiters of the text a request asks to rewrite (Glimpse's cleanup and Edit Mode turns). Drafts come from
/// inside the first block found, so instructions around it are never drafted; without one, the whole user turn.
const SOURCE_TAGS: [(&str, &str); 2] = [("<transcript>\n", "\n</transcript>"), ("<text>\n", "\n</text>")];
/// Words a cleanup capitalizes anywhere (drafts get them capitalized).
const CAPITALIZED: [&[u8]; 18] = [
    b"monday", b"tuesday", b"wednesday", b"thursday", b"friday", b"saturday", b"sunday", b"january", b"february",
    b"march", b"april", b"june", b"july", b"august", b"september", b"october", b"november", b"december",
];
/// Contractions a transcript may spell without the apostrophe.
const CONTRACTIONS: [(&str, &str); 22] = [
    ("dont", "don't"), ("doesnt", "doesn't"), ("didnt", "didn't"), ("isnt", "isn't"), ("wasnt", "wasn't"),
    ("arent", "aren't"), ("werent", "weren't"), ("wont", "won't"), ("cant", "can't"), ("couldnt", "couldn't"),
    ("wouldnt", "wouldn't"), ("shouldnt", "shouldn't"), ("havent", "haven't"), ("hasnt", "hasn't"),
    ("im", "I'm"), ("ive", "I've"), ("youre", "you're"), ("theyre", "they're"), ("thats", "that's"),
    ("whats", "what's"), ("theres", "there's"), ("lets", "let's"),
];
/// Hesitation tokens the cleanup drops; they are left out of the draft source.
const FILLERS: [&str; 12] = [" um", " uh", " Um", " Uh", " umm", " uhm", " erm", " hmm", "um", "uh", "Um", "Uh"];

/// Encodes a deferred `accept` (see `Model::defer_accept`) into `cmd`.
fn encode_pending(pending: &mut Option<(usize, usize)>, text: &mut Text, wb: &gpu::Buffer, cmd: &mut gpu::Cmd) {
    if let Some((t, n)) = pending.take() {
        text.encode_accept(cmd, wb, t, n);
    }
}

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

/// Prefix states kept in memory (and files kept in `Model::prefix_dir`): Glimpse's cleanup prompt, Edit Mode
/// and one writing mode. About 34 MB each on the 2B with a 574-token prompt, 90 MB on the 4B.
const PREFIX_SLOTS: usize = 3;
/// Saved prefix file format; bump when the layout or what the state means changes.
const PREFIX_MAGIC: &[u8; 8] = b"BNSPFX01";
const FNV: u64 = 0xcbf29ce484222325;

/// State after a prefix (compact: KV sized to the prefix), for one adapter.
struct Prefix {
    ids: Vec<u32>,
    adapter: u64,
    snap: State,
}

/// FNV-1a style hash over 8-byte words (fast enough for a 31 MB adapter), then the tail bytes.
fn hash_bytes(mut h: u64, b: &[u8]) -> u64 {
    let mut words = b.chunks_exact(8);
    for w in &mut words {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x100000001b3);
    }
    for &c in words.remainder() {
        h = (h ^ c as u64).wrapping_mul(0x100000001b3);
    }
    h ^ b.len() as u64
}

/// Identity of an adapter as loaded: its file, the PEFT config beside it (alpha / r) and the scale. Never 0.
fn adapter_id(path: &Path, scale: f32) -> Result<u64, String> {
    let mut h = hash_bytes(FNV, &std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?);
    if let Ok(cfg) = std::fs::read(path.parent().unwrap_or(Path::new(".")).join("adapter_config.json")) {
        h = hash_bytes(h, &cfg);
    }
    Ok(hash_bytes(h, &scale.to_le_bytes()) | 1)
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
    /// Cached prefix states, most recently used first.
    prefixes: Vec<Prefix>,
    /// Identity of the installed adapter (`adapter_id`), 0 without one.
    adapter: u64,
    /// Where prefix states are saved and looked up across processes; None keeps them in memory only.
    pub prefix_dir: Option<std::path::PathBuf>,
    dir: std::path::PathBuf,
    /// Prompt-lookup speculation for generate() (on by default; draft length chosen per round by `Policy`, up
    /// to `draft_max`; output is token-identical to plain greedy); None decodes one token per step, pipelined.
    pub spec: Option<Spec>,
    /// Deferred accept (t, n) of the last verify.
    pending: Option<(usize, usize)>,
    /// Benchmarks only: keep decoding past EOS.
    pub ignore_eos: bool,
    /// Draft matching key per vocabulary id (`token_key`), built on first use.
    keys: std::sync::OnceLock<std::sync::Arc<Vec<u64>>>,
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
            prefixes: Vec::new(),
            adapter: 0,
            prefix_dir: None,
            dir: dir.to_path_buf(),
            spec: Some(Spec::default()),
            pending: None,
            ignore_eos: false,
            keys: std::sync::OnceLock::new(),
        })
    }

    /// Applies a LoRA adapter (safetensors, see lora.rs) with B scaled by `scale`; None or scale 0 runs the base.
    /// Drops cached prefix states made with a different adapter.
    pub fn set_lora(&mut self, path: Option<&Path>, scale: f32) -> Result<(), String> {
        let (l, id) = match path {
            Some(p) if scale != 0.0 => (Some(lora::load(p, scale, &self.text, &self.gpu)?), adapter_id(p, scale)?),
            _ => (None, 0),
        };
        self.text.set_lora(&self.gpu, l);
        self.adapter = id;
        self.prefixes.retain(|p| p.adapter == id);
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

    /// Restores the cached state after `prefix` (token ids) from memory or `prefix_dir`, or prefills and
    /// caches it. Returns the number of tokens served from a cache.
    pub fn start_prefix(&mut self, prefix: &[u32]) -> usize {
        self.pending = None;
        if prefix.is_empty() {
            self.reset();
            return 0;
        }
        let hit = self.prefixes.iter().position(|p| p.ids == prefix && p.adapter == self.adapter);
        let hit = hit.map(|i| self.prefixes.remove(i)).or_else(|| self.load_prefix(prefix));
        if let Some(p) = hit {
            let mut cmd = self.gpu.begin();
            self.text.encode_copy_state(&mut cmd, &p.snap, &self.text.state);
            cmd.finish();
            self.text.state.len = p.snap.len;
            self.text.state.pos = p.snap.pos;
            self.prefixes.insert(0, p);
            return prefix.len();
        }
        self.reset();
        self.prefill(prefix, &Self::text_positions(0, prefix.len()), &[]);
        let mut snap = self.new_snapshot(self.text.state.len);
        snap.len = self.text.state.len;
        snap.pos = self.text.state.pos;
        let mut cmd = self.gpu.begin();
        self.text.encode_copy_state(&mut cmd, &self.text.state, &snap);
        cmd.finish();
        let p = Prefix { ids: prefix.to_vec(), adapter: self.adapter, snap };
        self.save_prefix(&p);
        self.prefixes.insert(0, p);
        self.prefixes.truncate(PREFIX_SLOTS);
        0
    }

    /// A state sized for `len` positions.
    fn new_snapshot(&self, len: usize) -> State {
        State::new(&self.gpu, &self.text.cfg, len, self.text.n_gdn, self.text.n_attn)
    }

    /// Everything a saved prefix state depends on besides its token ids: format, kernels, executable, GPU,
    /// model files and adapter. None when a part cannot be read (nothing is saved or loaded then).
    fn prefix_identity(&self) -> Option<Vec<u8>> {
        let stamp = |p: &Path| -> Option<[u64; 2]> {
            let m = std::fs::metadata(p).ok()?;
            let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
            Some([m.len(), t.as_nanos() as u64])
        };
        let mut id = PREFIX_MAGIC.to_vec();
        let mut words = vec![hash_bytes(FNV, gpu::KERNEL_BLOB), self.adapter];
        words.extend(stamp(&std::env::current_exe().ok()?)?);
        words.extend(stamp(&self.dir.join("model.safetensors"))?);
        words.push(hash_bytes(FNV, &std::fs::read(self.dir.join("config.json")).ok()?));
        let c = &self.text.cfg;
        words.extend([self.text.n_gdn, self.text.n_attn, c.hidden, c.kv_width(), c.conv_dim()].map(|v| v as u64));
        id.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        id.extend(self.gpu.device_name().bytes());
        Some(id)
    }

    fn prefix_path(&self, identity: &[u8], ids: &[u32]) -> Option<std::path::PathBuf> {
        let h = hash_bytes(hash_bytes(FNV, identity), &ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
        Some(self.prefix_dir.as_ref()?.join(format!("{h:016x}.prefix")))
    }

    /// Reads a saved state for `ids` made by this exact model, adapter, executable and GPU, if there is one.
    fn load_prefix(&self, ids: &[u32]) -> Option<Prefix> {
        use std::io::Read;
        self.prefix_dir.as_ref()?;
        let identity = self.prefix_identity()?;
        let mut f = std::fs::File::open(self.prefix_path(&identity, ids)?).ok()?;
        let mut head = vec![0u8; identity.len() + 4 * ids.len() + 16];
        f.read_exact(&mut head).ok()?;
        let want: Vec<u8> = identity.iter().copied().chain(ids.iter().flat_map(|t| t.to_le_bytes())).collect();
        if head[..want.len()] != want[..] {
            return None;
        }
        let len = u64::from_le_bytes(head[want.len()..want.len() + 8].try_into().ok()?) as usize;
        let pos = i64::from_le_bytes(head[want.len() + 8..].try_into().ok()?) as i32;
        if len != ids.len() {
            return None;
        }
        let mut snap = self.new_snapshot(len);
        for (buf, n) in self.text.state_parts(&snap) {
            let mut part = vec![0u8; n];
            f.read_exact(&mut part).ok()?;
            gpu::write(buf, 0, &part);
        }
        if f.read(&mut [0u8]).ok()? != 0 {
            return None;
        }
        (snap.len, snap.pos) = (len, pos);
        Some(Prefix { ids: ids.to_vec(), adapter: self.adapter, snap })
    }

    /// Writes `p` to `prefix_dir` on a background thread (temporary file, then rename) and keeps the newest
    /// PREFIX_SLOTS files there.
    fn save_prefix(&self, p: &Prefix) {
        let Some(identity) = self.prefix_identity() else { return };
        let (Some(dir), Some(path)) = (self.prefix_dir.clone(), self.prefix_path(&identity, &p.ids)) else { return };
        let mut data = identity;
        data.extend(p.ids.iter().flat_map(|t| t.to_le_bytes()));
        data.extend((p.snap.len as u64).to_le_bytes());
        data.extend((p.snap.pos as i64).to_le_bytes());
        for (buf, n) in self.text.state_parts(&p.snap) {
            data.extend(gpu::read(buf, 0, n));
        }
        std::thread::spawn(move || {
            let tmp = path.with_extension("tmp");
            if std::fs::create_dir_all(&dir).is_err() || std::fs::write(&tmp, &data).is_err() || std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            let Ok(entries) = std::fs::read_dir(&dir) else { return };
            let mut files: Vec<_> = entries
                .filter_map(|e| {
                    let e = e.ok()?;
                    let name = e.file_name();
                    name.to_str()?.ends_with(".prefix").then(|| Some((e.metadata().ok()?.modified().ok()?, e.path())))?
                })
                .collect();
            files.sort_by(|a, b| b.0.cmp(&a.0));
            for (_, old) in files.into_iter().skip(PREFIX_SLOTS) {
                let _ = std::fs::remove_file(old);
            }
        });
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

    /// Drops the prefix states held in memory (saved files stay).
    pub fn clear_prefix_cache(&mut self) {
        self.prefixes.clear();
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

    /// What a cleanup does to the words of a lowercase transcript, applied to a draft word by word: " i"
    /// becomes " I", a word after a sentence end or in CAPITALIZED gets a capital, CONTRACTIONS get their
    /// apostrophe. Changed words are re-tokenized; the draft keeps at most its length.
    fn normalize_draft(&self, draft: &mut Vec<u32>, last: u32) {
        let k = draft.len();
        let mut out: Vec<u32> = Vec::with_capacity(k + 4);
        let mut prev_end = self.tok.token_bytes(last).trim_ascii_end().last().is_some_and(|c| b".?!".contains(c));
        let mut i = 0;
        while i < draft.len() && out.len() < k {
            // a word: a token starting with a space, plus the tokens that continue it
            let mut j = i + 1;
            while j < draft.len() && !self.tok.token_bytes(draft[j]).first().is_some_and(|c| !c.is_ascii_alphabetic() && *c != b'\'') {
                j += 1;
            }
            let w: Vec<u8> = draft[i..j].iter().flat_map(|&d| self.tok.token_bytes(d).to_vec()).collect();
            let mut v = w.clone();
            if v.len() > 1 && v[0] == b' ' && v[1..].iter().all(|c| c.is_ascii_lowercase()) {
                let bare = &w[1..];
                if let Some((_, to)) = CONTRACTIONS.iter().find(|(from, _)| from.as_bytes() == bare) {
                    v = format!(" {to}").into_bytes();
                } else if bare == b"i" || prev_end || CAPITALIZED.contains(&bare) {
                    v[1] = v[1].to_ascii_uppercase();
                }
            }
            match (v != w).then(|| String::from_utf8(v).ok()).flatten() {
                Some(text) => out.extend(self.tok.encode(&text)),
                None => out.extend_from_slice(&draft[i..j]),
            }
            prev_end = w.trim_ascii_end().last().is_some_and(|c| b".?!".contains(c));
            i = j;
        }
        out.truncate(k);
        *draft = out;
    }

    /// Draft source for a user turn: the tokens inside the first SOURCE_TAGS block (all of them if there is
    /// none), fillers removed.
    fn draft_source(&self, user: &[u32], keys: &[u64]) -> Source {
        let mut text = Vec::new();
        let mut ends = Vec::with_capacity(user.len());
        for &id in user {
            text.extend_from_slice(self.tok.token_bytes(id));
            ends.push(text.len());
        }
        let find = |pat: &str, from: usize| {
            text.get(from..)?.windows(pat.len()).position(|w| w == pat.as_bytes()).map(|i| i + from)
        };
        let (lo, hi) = SOURCE_TAGS
            .iter()
            .find_map(|(open, close)| {
                let a = find(open, 0)? + open.len();
                Some((a, find(close, a)?))
            })
            .unwrap_or((0, text.len()));
        let fillers: Vec<u32> = FILLERS
            .iter()
            .filter_map(|f| match self.tok.encode(f).as_slice() {
                [id] => Some(*id),
                _ => None,
            })
            .collect();
        let ids: Vec<u32> = (0..user.len())
            .filter(|&i| ends[i] > lo && ends[i] - self.tok.token_bytes(user[i]).len() >= lo && ends[i] <= hi)
            .map(|i| user[i])
            .filter(|id| !fillers.contains(id))
            .collect();
        let words = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| Some((keys.get(id as usize).copied().filter(|&k| k & 1 != 0)?, i)))
            .collect();
        Source { ids, words }
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
        let cost = if self.text.cfg.hidden <= 2048 { &gpu::VERIFY_COST_2B } else { &gpu::VERIFY_COST_4B };
        let mut pol = Policy::new(cost, sp.draft_max);
        let keys = self.keys.get_or_init(|| {
            std::sync::Arc::new((0..self.tok.vocab_len() as u32).map(|id| token_key(self.tok.token_bytes(id))).collect())
        });
        let keys = std::sync::Arc::clone(keys);
        let src = self.draft_source(user, &keys);
        let mut seq = user.to_vec();
        seq.extend_from_slice(out);
        // where in `src` the last draft's accepted part ended; rounds in a row whose first draft token failed
        // at source position .0 (a draft from there is not tried a third time)
        let mut cursor = 0;
        let mut failed = (usize::MAX, 0);
        let nl = |id: u32| keys.get(id as usize).is_some_and(|k| k & 2 != 0);
        let src_nl = src.ids.iter().any(|&id| nl(id));
        // after a line break the source does not have, list markers or a new paragraph follow, not the draft
        let brk = |id: u32| nl(id) && !src_nl;
        let given_up = |f: (usize, usize), j: usize| f.0 == j && f.1 >= 2;
        // realized tokens and table cost of recent verify rounds (decayed like the policy's counts)
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
            let (mut draft, dpos) = lookup(&src, &seq, &keys, &Spec { draft_max: k, ..sp }, left - 1, cursor);
            if given_up(failed, dpos) || brk(last) {
                draft.clear();
            }
            if draft.is_empty() {
                // Plain decoding; shadow rounds (what lookup would have drafted vs what came out) update the
                // estimate, and speculation resumes once it clears the entry margin.
                let (mut next_start, mut sh_pos, mut sh, mut sh_src): (usize, usize, Vec<u32>, usize) =
                    (seq.len(), 0, Vec::new(), 0);
                let stop = |id: u32| {
                    seq.push(id);
                    let j = seq.len() - 1;
                    if !sh.is_empty() && j >= sh_pos {
                        let i = j - sh_pos;
                        if seq[j] != sh[i] {
                            pol.update(sh.len(), i);
                            cursor = sh_src + i;
                            sh.clear();
                            next_start = j + 1;
                        } else if i + 1 == sh.len() {
                            pol.update(sh.len(), sh.len());
                            cursor = sh_src + sh.len();
                            sh.clear();
                            next_start = j + 2;
                        }
                    }
                    if sh.is_empty() && seq.len() >= next_start {
                        (sh, sh_src) = lookup(&src, &seq, &keys, &sp, usize::MAX, cursor);
                        sh_pos = seq.len();
                        next_start = seq.len() + 1;
                    }
                    // switch back only when lookup has a draft for the next round
                    pol.choose(sp.draft_max, ENTER_GAIN).0 > 0 && !brk(id) && {
                        let (d, j) = lookup(&src, &seq, &keys, &sp, usize::MAX, cursor);
                        !d.is_empty() && !given_up(failed, j)
                    }
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
            self.normalize_draft(&mut draft, last);
            let new = {
                let mut ids = vec![last];
                ids.extend_from_slice(&draft);
                let pred = self.verify(&ids);
                let m = draft.iter().zip(&pred).take_while(|(d, p)| d == p).count();
                self.defer_accept(ids.len(), m + 1);
                pol.update(draft.len(), m);
                cursor = dpos + m;
                failed = match (m, failed) {
                    (0, (p, n)) if p == dpos => (p, n + 1),
                    (0, _) => (dpos, 1 + brk(pred[m]) as usize),
                    _ => (usize::MAX, 0),
                };
                got = got * Policy::DECAY + (m + 1) as f64;
                spent = spent * Policy::DECAY + cost[draft.len()];
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
