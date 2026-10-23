//! The public API over libllama and ggml (Windows): PrismML's Q2_0 GGUF on the best Vulkan GPU. Same requests
//! and greedy output as the Metal runtime: our tokenizer and Qwen3.5 template, prompt-lookup speculation,
//! cached system prompt states and the same LoRA adapter files. No CPU fallback: a 50-word cleanup takes
//! over 20 s on a 24-core Xeon.

use transcribe_cpp_sys as _;

mod adapter;
mod spec;

use std::ffi::{CString, c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use super::tokenizer::Tokenizer;
use super::{
    Availability, BonsaiError, BonsaiErrorKind, CONTEXT, GenerateOptions, Generation, Stats,
    StopReason,
};
use spec::{ENTER_GAIN, GIVE_UP_ROUNDS, KEEP_GAIN, Policy};

#[repr(C)]
struct GbModel {
    _private: [u8; 0],
}

#[repr(C)]
struct GbCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn gb_init(verbose: c_int);
    fn gb_device_count() -> c_int;
    fn gb_device_info(
        i: c_int,
        kind: *mut c_int,
        name: *mut c_char,
        len: c_int,
        free: *mut u64,
        total: *mut u64,
    ) -> c_int;
    fn gb_model_load(
        path: *const c_char,
        device: c_int,
        err: *mut c_char,
        len: c_int,
    ) -> *mut GbModel;
    fn gb_model_set_lora(
        m: *mut GbModel,
        path: *const c_char,
        err: *mut c_char,
        len: c_int,
    ) -> c_int;
    fn gb_model_free(m: *mut GbModel);
    fn gb_ctx_new(
        m: *mut GbModel,
        n_ctx: c_int,
        n_batch: c_int,
        n_out_max: c_int,
        n_rs: c_int,
        err: *mut c_char,
        len: c_int,
    ) -> *mut GbCtx;
    fn gb_ctx_free(c: *mut GbCtx);
    fn gb_decode(
        c: *mut GbCtx,
        ids: *const i32,
        n: c_int,
        pos0: c_int,
        n_out: c_int,
        out: *mut i32,
    ) -> c_int;
    fn gb_seq_rm(c: *mut GbCtx, p0: c_int) -> c_int;
    fn gb_clear(c: *mut GbCtx);
    fn gb_state_size(c: *mut GbCtx, flags: u32) -> u64;
    fn gb_state_get(c: *mut GbCtx, dst: *mut u8, size: u64, flags: u32) -> u64;
    fn gb_state_set(c: *mut GbCtx, src: *const u8, size: u64, flags: u32) -> u64;
}

/// The model file in a model directory on this backend.
pub const WEIGHTS: &str = "model.gguf";
/// Saved system prompt states, inside the model directory.
const PREFIX_DIR: &str = "prefix-cache";
const PREFIX_SLOTS: usize = 3;
const PREFIX_MAGIC: &[u8; 8] = b"BNSGGF01";
const EOS: [u32; 2] = [248046, 248044];
/// Prefill chunk.
const BATCH: usize = 512;
/// Recurrent state snapshots for rolling back rejected drafts.
const ROLLBACK: usize = spec::DRAFT_MAX;
/// Positions kept free after the prompt, as on Metal, so both backends fail and cap at the same lengths.
const HEADROOM: usize = 16;
const STATE_PARTIAL: u32 = 1;
const STATE_ON_DEVICE: u32 = 2;
/// Verify cost in decode steps for 1..=8 rows (the last token plus drafts), RTX 4000 Ada over Vulkan with
/// the adapter. Rows past DRAFT_MAX + 1 are unused.
const VERIFY_COST: [f64; 8] = [1.0, 1.14, 1.32, 1.49, 1.75, 1.92, 2.2, 2.41];

const SYSTEM: (&str, &str) = ("<|im_start|>system\n", "<|im_end|>\n");
const USER: (&str, &str) = ("<|im_start|>user\n", "<|im_end|>\n");
const ASSISTANT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn verbose() -> bool {
    std::env::var_os("BONSAI_LOG").is_some()
}

/// The GPU to use, as a gb_model_load index: the first discrete one, else the first integrated one. None
/// without a Vulkan GPU.
fn device() -> Option<c_int> {
    static DEVICE: OnceLock<Option<c_int>> = OnceLock::new();
    *DEVICE.get_or_init(|| {
        unsafe { gb_init(verbose() as c_int) };
        let count = unsafe { gb_device_count() };
        if verbose() {
            for i in 0..count {
                let (mut kind, mut free, mut total) = (0, 0, 0);
                let mut name = [0 as c_char; 256];
                unsafe {
                    gb_device_info(i, &mut kind, name.as_mut_ptr(), 256, &mut free, &mut total)
                };
                let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
                eprintln!(
                    "bonsai: GPU {i}: {name} (kind {kind}, {} MB free)",
                    free >> 20
                );
            }
        }
        (count > 0).then_some(0)
    })
}

pub fn availability() -> Availability {
    if device().is_some() {
        Availability::Available
    } else {
        Availability::Unsupported
    }
}

fn load_err(message: impl Into<String>) -> BonsaiError {
    BonsaiError::new(BonsaiErrorKind::Load, message)
}

fn runtime_err(message: impl Into<String>) -> BonsaiError {
    BonsaiError::new(BonsaiErrorKind::Runtime, message)
}

fn c_path(path: &Path) -> Result<CString, BonsaiError> {
    let s = path
        .to_str()
        .ok_or_else(|| load_err(format!("{}: not UTF-8", path.display())))?;
    CString::new(s).map_err(|_| load_err(format!("{}: bad path", path.display())))
}

fn err_text(buf: &[c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Weights on the device, plus the adapter.
struct Weights {
    ptr: *mut GbModel,
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe { gb_model_free(self.ptr) };
    }
}

/// KV cache, recurrent state and compute buffers.
struct Context {
    ptr: *mut GbCtx,
    /// The last system prompt: its token ids and its recurrent state, kept on the device. Its KV cells stay in
    /// the cache, since requests only add cells after it.
    checkpoint: Option<(Vec<u32>, Vec<u8>)>,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { gb_ctx_free(self.ptr) };
    }
}

impl Context {
    /// Decodes `ids` at `pos`, returning the argmax after each of the last `n_out` tokens.
    fn decode(&mut self, ids: &[u32], pos: usize, n_out: usize) -> Result<Vec<u32>, BonsaiError> {
        let mut out = vec![0i32; n_out];
        let rc = unsafe {
            gb_decode(
                self.ptr,
                ids.as_ptr().cast(),
                ids.len() as c_int,
                pos as c_int,
                n_out as c_int,
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(runtime_err(format!("decode failed ({rc})")));
        }
        Ok(out.into_iter().map(|id| id as u32).collect())
    }

    fn truncate(&mut self, pos: usize) -> Result<(), BonsaiError> {
        if unsafe { gb_seq_rm(self.ptr, pos as c_int) } == 0 {
            return Err(runtime_err("state rollback failed"));
        }
        Ok(())
    }

    fn clear(&mut self) {
        unsafe { gb_clear(self.ptr) };
        self.checkpoint = None;
    }

    fn get_state(&mut self, flags: u32) -> Vec<u8> {
        let size = unsafe { gb_state_size(self.ptr, flags) } as usize;
        let mut buf = vec![0u8; size];
        let n = unsafe { gb_state_get(self.ptr, buf.as_mut_ptr(), size as u64, flags) } as usize;
        buf.truncate(n);
        buf
    }

    fn set_state(&mut self, data: &[u8], flags: u32) -> bool {
        let n = unsafe { gb_state_set(self.ptr, data.as_ptr(), data.len() as u64, flags) };
        n == data.len() as u64
    }
}

/// A system prompt's state: the full sequence state, in host memory.
struct Prefix {
    ids: Vec<u32>,
    state: Vec<u8>,
}

struct Model {
    weights: Option<Weights>,
    ctx: Option<Context>,
    tok: Tokenizer,
    /// Matching keys per token for speculation, built on first use.
    keys: Option<Arc<Vec<u64>>>,
    prefixes: Vec<Prefix>,
    adapter_id: u64,
}

// The pointers are only used under the handle's mutex.
unsafe impl Send for Model {}

pub struct Handle {
    dir: PathBuf,
    adapter: Option<PathBuf>,
    device: c_int,
    state: Mutex<Option<Model>>,
}

fn heads(dir: &Path) -> Result<adapter::Heads, BonsaiError> {
    let json = std::fs::read(dir.join("config.json"))
        .map_err(|e| load_err(format!("config.json: {e}")))?;
    let v: serde_json::Value =
        serde_json::from_slice(&json).map_err(|e| load_err(format!("config.json: {e}")))?;
    let t = &v["text_config"];
    let get = |k: &str| {
        t[k].as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| load_err(format!("config.json: {k} missing")))
    };
    Ok(adapter::Heads {
        k_heads: get("linear_num_key_heads")?,
        v_heads: get("linear_num_value_heads")?,
        v_dim: get("linear_value_head_dim")?,
        k_dim: get("linear_key_head_dim")?,
    })
}

const FNV: u64 = 0xcbf29ce484222325;

fn hash_bytes(mut h: u64, b: &[u8]) -> u64 {
    for &c in b {
        h = (h ^ c as u64).wrapping_mul(0x100000001b3);
    }
    h ^ b.len() as u64
}

impl Handle {
    pub fn load(dir: &Path, adapter: Option<&Path>) -> Result<Handle, BonsaiError> {
        let Some(device) = device() else {
            return Err(BonsaiError::new(
                BonsaiErrorKind::Unsupported,
                "Bonsai needs a Vulkan GPU",
            ));
        };
        let handle = Handle {
            dir: dir.to_path_buf(),
            adapter: adapter.map(Path::to_path_buf),
            device,
            state: Mutex::new(None),
        };
        let model = handle.load_model()?;
        *handle.lock() = Some(model);
        Ok(handle)
    }

    fn lock(&self) -> MutexGuard<'_, Option<Model>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load_model(&self) -> Result<Model, BonsaiError> {
        let tok = Tokenizer::load(
            &self.dir.join("tokenizer.json"),
            &self.dir.join("tokenizer.bonsai"),
        )
        .map_err(load_err)?;
        let adapter_id = match &self.adapter {
            Some(p) => hash_bytes(
                FNV,
                &std::fs::read(p).map_err(|e| load_err(format!("{}: {e}", p.display())))?,
            ),
            None => 0,
        };
        let mut model = Model {
            weights: None,
            ctx: None,
            tok,
            keys: None,
            prefixes: Vec::new(),
            adapter_id,
        };
        self.wake(&mut model)?;
        Ok(model)
    }

    /// Loads the weights, the adapter and a fresh context.
    fn wake(&self, model: &mut Model) -> Result<(), BonsaiError> {
        let t = Instant::now();
        let mut err = [0 as c_char; 512];
        if model.weights.is_none() {
            let path = c_path(&self.dir.join(WEIGHTS))?;
            let ptr = unsafe {
                gb_model_load(
                    path.as_ptr(),
                    self.device,
                    err.as_mut_ptr(),
                    err.len() as c_int,
                )
            };
            if ptr.is_null() {
                return Err(load_err(err_text(&err)));
            }
            let weights = Weights { ptr };
            if let Some(adapter) = &self.adapter {
                let out = std::env::temp_dir().join(format!(
                    "glimpse-bonsai-{}-{}.gguf",
                    std::process::id(),
                    model.adapter_id
                ));
                let converted =
                    adapter::convert(adapter, 1.0, heads(&self.dir)?, &out).map_err(load_err);
                let rc = converted.and_then(|_| {
                    let p = c_path(&out)?;
                    Ok(unsafe {
                        gb_model_set_lora(
                            weights.ptr,
                            p.as_ptr(),
                            err.as_mut_ptr(),
                            err.len() as c_int,
                        )
                    })
                });
                let _ = std::fs::remove_file(&out);
                if rc? != 0 {
                    return Err(load_err(err_text(&err)));
                }
            }
            model.weights = Some(weights);
        }
        if model.ctx.is_none() {
            let w = model.weights.as_ref().unwrap();
            let ptr = unsafe {
                gb_ctx_new(
                    w.ptr,
                    CONTEXT as c_int,
                    BATCH as c_int,
                    (spec::DRAFT_MAX + 1) as c_int,
                    ROLLBACK as c_int,
                    err.as_mut_ptr(),
                    err.len() as c_int,
                )
            };
            if ptr.is_null() {
                return Err(load_err(err_text(&err)));
            }
            let mut ctx = Context {
                ptr,
                checkpoint: None,
            };
            static WARM: AtomicBool = AtomicBool::new(false);
            if !WARM.load(Ordering::Relaxed) {
                warm_up(&mut ctx, &model.tok)?;
                WARM.store(true, Ordering::Relaxed);
            }
            model.ctx = Some(ctx);
        }
        if verbose() {
            eprintln!("bonsai: wake {:?}", t.elapsed());
        }
        Ok(())
    }

    /// Runs `f` on the loaded model, rebuilding what release_idle freed. A failure drops the model so the
    /// next call loads it again.
    fn with_model<T>(
        &self,
        f: impl FnOnce(&mut Model) -> Result<T, BonsaiError>,
    ) -> Result<T, BonsaiError> {
        let mut state = self.lock();
        if state.is_none() {
            *state = Some(self.load_model()?);
        }
        let model = state.as_mut().unwrap();
        self.wake(model)?;
        let result = f(model);
        if result
            .as_ref()
            .is_err_and(|e| e.kind == BonsaiErrorKind::Runtime)
        {
            *state = None;
        }
        result
    }

    pub fn prewarm(&self, system: &str) -> Result<(), BonsaiError> {
        self.with_model(|m| m.start_system(system, &self.dir).map(|_| ()))
    }

    pub fn generate(
        &self,
        system: &str,
        user: &str,
        options: &GenerateOptions,
        cancel: Option<&AtomicBool>,
    ) -> Result<Generation, BonsaiError> {
        self.with_model(|m| m.generate(system, user, options, cancel, &self.dir))
    }

    /// Frees the device memory (weights, KV cache, recurrent state). The tokenizer and the cached prompt
    /// states stay in host memory.
    pub fn release_idle(&self) {
        let mut state = self.lock();
        if let Some(m) = state.as_mut() {
            m.ctx = None;
            m.weights = None;
        }
    }

    pub fn is_idle(&self) -> bool {
        self.lock().as_ref().is_some_and(|m| m.weights.is_none())
    }
}

impl Model {
    fn ctx(&mut self) -> &mut Context {
        self.ctx.as_mut().expect("context")
    }

    /// Restores or builds the state after the system turn. Returns the prefix length.
    fn start_system(&mut self, system: &str, dir: &Path) -> Result<usize, BonsaiError> {
        let ids = if system.trim().is_empty() {
            Vec::new()
        } else {
            self.tok
                .encode(&format!("{}{}{}", SYSTEM.0, system.trim(), SYSTEM.1))
        };
        if ids.len() + HEADROOM + 1 > CONTEXT {
            return Err(BonsaiError::new(
                BonsaiErrorKind::ContextTooLarge,
                format!(
                    "prompt is {} tokens, context holds {CONTEXT} with room for output",
                    ids.len()
                ),
            ));
        }
        if ids.is_empty() {
            self.ctx().clear();
            return Ok(0);
        }
        let n = ids.len();
        let ctx = self.ctx.as_mut().unwrap();
        if let Some((cached, state)) = &ctx.checkpoint
            && *cached == ids
        {
            let state = state.clone();
            if ctx.set_state(&state, STATE_PARTIAL | STATE_ON_DEVICE) && ctx.truncate(n).is_ok() {
                if let Some(i) = self.prefixes.iter().position(|p| p.ids == ids) {
                    let p = self.prefixes.remove(i);
                    self.prefixes.insert(0, p);
                }
                return Ok(n);
            }
        }
        let hit = self.prefixes.iter().position(|p| p.ids == ids);
        let prefix = match hit {
            Some(i) => Some(self.prefixes.remove(i)),
            None => load_prefix(dir, &ids, self.adapter_id),
        };
        let ctx = self.ctx.as_mut().unwrap();
        let restored = prefix.as_ref().is_some_and(|p| {
            ctx.clear();
            ctx.set_state(&p.state, 0)
        });
        let prefix = if restored {
            prefix.unwrap()
        } else {
            ctx.clear();
            ctx.decode(&ids, 0, 0)?;
            let p = Prefix {
                ids: ids.clone(),
                state: ctx.get_state(0),
            };
            save_prefix(dir, &p, self.adapter_id);
            p
        };
        let state = ctx.get_state(STATE_PARTIAL | STATE_ON_DEVICE);
        ctx.checkpoint = (!state.is_empty()).then(|| (ids.clone(), state));
        self.prefixes.insert(0, prefix);
        self.prefixes.truncate(PREFIX_SLOTS);
        Ok(n)
    }

    fn generate(
        &mut self,
        system: &str,
        user: &str,
        options: &GenerateOptions,
        cancel: Option<&AtomicBool>,
        dir: &Path,
    ) -> Result<Generation, BonsaiError> {
        let t0 = Instant::now();
        let prefix = self.start_system(system, dir)?;
        let raw = self
            .tok
            .encode(&format!("{}{}{}{}", USER.0, user.trim(), USER.1, ASSISTANT));
        let used = prefix + raw.len();
        if used + HEADROOM + 1 > CONTEXT {
            return Err(BonsaiError::new(
                BonsaiErrorKind::ContextTooLarge,
                format!("prompt is {used} tokens, context holds {CONTEXT} with room for output"),
            ));
        }
        let max_tokens = options.max_tokens.min(CONTEXT - used - HEADROOM);
        let mut st = Stats {
            cached_tokens: prefix,
            prompt_tokens: raw.len(),
            ..Default::default()
        };
        let tp = Instant::now();
        let first = self.ctx().decode(&raw, prefix, 1)?[0];
        st.prefill = tp.elapsed();
        st.first_token = t0.elapsed();
        let td = Instant::now();
        let mut out = Vec::new();
        let stop_eos = options.stop_at_eos;
        if max_tokens > 0 && !(stop_eos && EOS.contains(&first)) {
            out.push(first);
            let mut d = Decoder {
                m: self,
                pos: used,
                stop_eos,
                cancel,
                max_tokens,
            };
            if options.speculate {
                d.speculative(&raw, &mut out, &mut st)?;
            } else {
                d.plain(&mut out, |_| false)?;
            }
        }
        st.decode = td.elapsed();
        st.generated_tokens = out.len();
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(BonsaiError::new(BonsaiErrorKind::Cancelled, "cancelled"));
        }
        let stop = if out.len() < max_tokens {
            StopReason::Eos
        } else if max_tokens < options.max_tokens {
            StopReason::ContextFull
        } else {
            StopReason::MaxTokens
        };
        Ok(Generation {
            text: self.tok.decode(&out).trim().to_string(),
            stop,
            stats: st,
        })
    }
}

/// Runs every batch shape a request uses once (prefill chunks, decode, verify), so the device pipelines are
/// built at load and not during the first request. Once per process; the pipelines outlive contexts.
fn warm_up(ctx: &mut Context, tok: &Tokenizer) -> Result<(), BonsaiError> {
    let t = Instant::now();
    let word = tok.encode(" the");
    let ids: Vec<u32> = word.iter().copied().cycle().take(BATCH).collect();
    ctx.clear();
    ctx.decode(&ids, 0, 1)?;
    let mut pos = BATCH;
    for n in [100, 68, 49, 20, 9] {
        ctx.decode(&ids[..n], pos, 1)?;
        pos += n;
    }
    for k in 1..=spec::DRAFT_MAX + 1 {
        ctx.decode(&ids[..k], pos, k)?;
        if k > 1 {
            ctx.truncate(pos + 1)?;
        }
        pos += 1;
    }
    ctx.clear();
    if verbose() {
        eprintln!("bonsai: warm-up {:?}", t.elapsed());
    }
    Ok(())
}

fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}

/// Greedy decoding after the prefill. `out` ends with a token that is not fed yet; `pos` is where it goes.
struct Decoder<'a> {
    m: &'a mut Model,
    pos: usize,
    stop_eos: bool,
    cancel: Option<&'a AtomicBool>,
    max_tokens: usize,
}

impl Decoder<'_> {
    fn is_stop(&self, id: u32) -> bool {
        self.stop_eos && EOS.contains(&id)
    }

    /// One token per step. `switch(id)` is called for every emitted token; true returns Ok(true).
    /// Ok(false): EOS, the token cap or cancel.
    fn plain(
        &mut self,
        out: &mut Vec<u32>,
        mut switch: impl FnMut(u32) -> bool,
    ) -> Result<bool, BonsaiError> {
        while out.len() < self.max_tokens {
            if cancelled(self.cancel) {
                return Ok(false);
            }
            let last = *out.last().unwrap();
            let id = self.m.ctx().decode(&[last], self.pos, 1)?[0];
            self.pos += 1;
            if self.is_stop(id) {
                return Ok(false);
            }
            out.push(id);
            if out.len() < self.max_tokens && switch(id) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Prompt-lookup speculation with the adaptive policy, as runtime/mod.rs `decode_spec`.
    fn speculative(
        &mut self,
        user: &[u32],
        out: &mut Vec<u32>,
        st: &mut Stats,
    ) -> Result<(), BonsaiError> {
        let mut pol = Policy::new(&VERIFY_COST, spec::DRAFT_MAX);
        let tok = &self.m.tok;
        let keys = Arc::clone(self.m.keys.get_or_insert_with(|| {
            Arc::new(
                (0..tok.vocab_len() as u32)
                    .map(|id| spec::token_key(tok.token_bytes(id)))
                    .collect(),
            )
        }));
        let src = spec::draft_source(&self.m.tok, user, &keys);
        let mut seq = user.to_vec();
        seq.extend_from_slice(out);
        let mut cursor = 0;
        let mut failed = (usize::MAX, 0);
        let nl = |id: u32| keys.get(id as usize).is_some_and(|k| k & 2 != 0);
        let src_nl = src.ids.iter().any(|&id| nl(id));
        let brk = |id: u32| nl(id) && !src_nl;
        let given_up = |f: (usize, usize), j: usize| f.0 == j && f.1 >= 2;
        let (mut got, mut spent, mut rounds) = (0.0, 0.0, 0);
        while out.len() < self.max_tokens {
            if cancelled(self.cancel) {
                return Ok(());
            }
            let last = *seq.last().unwrap();
            let left = self.max_tokens - out.len();
            if rounds >= GIVE_UP_ROUNDS && got < KEEP_GAIN * spent {
                self.plain(out, |_| false)?;
                return Ok(());
            }
            let k = pol.choose(left - 1, KEEP_GAIN);
            let (mut draft, dpos) = spec::lookup(&src, &seq, &keys, k.min(left - 1), cursor);
            if given_up(failed, dpos) || brk(last) {
                draft.clear();
            }
            if draft.is_empty() {
                // Plain decoding; shadow rounds score what lookup would have drafted, and speculation resumes
                // once the estimate clears the entry margin.
                let (mut next_start, mut sh_pos, mut sh, mut sh_src): (
                    usize,
                    usize,
                    Vec<u32>,
                    usize,
                ) = (seq.len(), 0, Vec::new(), 0);
                let switch = |id: u32| {
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
                        (sh, sh_src) = spec::lookup(&src, &seq, &keys, spec::DRAFT_MAX, cursor);
                        sh_pos = seq.len();
                        next_start = seq.len() + 1;
                    }
                    pol.choose(spec::DRAFT_MAX, ENTER_GAIN) > 0 && !brk(id) && {
                        let (d, j) = spec::lookup(&src, &seq, &keys, spec::DRAFT_MAX, cursor);
                        !d.is_empty() && !given_up(failed, j)
                    }
                };
                if !self.plain(out, switch)? {
                    return Ok(());
                }
                continue;
            }
            spec::normalize_draft(&self.m.tok, &mut draft, last);
            let mut ids = vec![last];
            ids.extend_from_slice(&draft);
            let pred = self.m.ctx().decode(&ids, self.pos, ids.len())?;
            let m = draft.iter().zip(&pred).take_while(|(d, p)| d == p).count();
            if m < draft.len() {
                self.m.ctx().truncate(self.pos + m + 1)?;
            }
            self.pos += m + 1;
            pol.update(draft.len(), m);
            cursor = dpos + m;
            failed = match (m, failed) {
                (0, (p, n)) if p == dpos => (p, n + 1),
                (0, _) => (dpos, 1 + brk(pred[m]) as usize),
                _ => (usize::MAX, 0),
            };
            got = got * Policy::DECAY + (m + 1) as f64;
            spent = spent * Policy::DECAY + VERIFY_COST[draft.len()];
            rounds += 1;
            st.drafted += draft.len();
            st.accepted += m;
            for &id in draft[..m].iter().chain(std::iter::once(&pred[m])) {
                if self.is_stop(id) || out.len() >= self.max_tokens {
                    return Ok(());
                }
                out.push(id);
                seq.push(id);
            }
        }
        Ok(())
    }
}

/// What a saved prefix state depends on besides its token ids: format, executable, model files, adapter.
fn prefix_identity(dir: &Path, adapter_id: u64) -> Option<Vec<u8>> {
    let stamp = |p: &Path| -> Option<[u64; 2]> {
        let m = std::fs::metadata(p).ok()?;
        let t = m
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        Some([m.len(), t.as_nanos() as u64])
    };
    let mut words = vec![adapter_id];
    words.extend(stamp(&std::env::current_exe().ok()?)?);
    words.extend(stamp(&dir.join(WEIGHTS))?);
    let mut id = PREFIX_MAGIC.to_vec();
    id.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    Some(id)
}

fn prefix_path(dir: &Path, identity: &[u8], ids: &[u32]) -> PathBuf {
    let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    let h = hash_bytes(hash_bytes(FNV, identity), &bytes);
    dir.join(PREFIX_DIR).join(format!("{h:016x}.prefix"))
}

/// A saved state for `ids` made by this executable, model and adapter, if there is one.
fn load_prefix(dir: &Path, ids: &[u32], adapter_id: u64) -> Option<Prefix> {
    let identity = prefix_identity(dir, adapter_id)?;
    let data = std::fs::read(prefix_path(dir, &identity, ids)).ok()?;
    let want: Vec<u8> = identity
        .iter()
        .copied()
        .chain(ids.iter().flat_map(|t| t.to_le_bytes()))
        .collect();
    let state = data.strip_prefix(&want[..])?;
    Some(Prefix {
        ids: ids.to_vec(),
        state: state.to_vec(),
    })
}

/// Writes `p` on a background thread (temporary file, then rename) and keeps the newest PREFIX_SLOTS files.
fn save_prefix(dir: &Path, p: &Prefix, adapter_id: u64) {
    let Some(identity) = prefix_identity(dir, adapter_id) else {
        return;
    };
    let path = prefix_path(dir, &identity, &p.ids);
    let folder = dir.join(PREFIX_DIR);
    let mut data = identity;
    data.extend(p.ids.iter().flat_map(|t| t.to_le_bytes()));
    data.extend_from_slice(&p.state);
    std::thread::spawn(move || {
        let tmp = path.with_extension("tmp");
        if std::fs::create_dir_all(&folder).is_err()
            || std::fs::write(&tmp, &data).is_err()
            || std::fs::rename(&tmp, &path).is_err()
        {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        let Ok(entries) = std::fs::read_dir(&folder) else {
            return;
        };
        let mut files: Vec<_> = entries
            .filter_map(|e| {
                let e = e.ok()?;
                let name = e.file_name();
                name.to_str()?
                    .ends_with(".prefix")
                    .then(|| Some((e.metadata().ok()?.modified().ok()?, e.path())))?
            })
            .collect();
        files.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, old) in files.into_iter().skip(PREFIX_SLOTS) {
            let _ = std::fs::remove_file(old);
        }
    });
}
