//! Glimpse-Language additions that need the runtime's private items. Ours, not synced from Glimpse-Lab.

use super::gpu::Gpu;
use super::model::{Config, Text, VMAX};
use super::tokenizer::Tokenizer;
use super::weights::Weights;
use super::{Model, Prefix, QWEN35, Spec, Stats};
use std::cell::Cell;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

thread_local! {
    static CANCEL: Cell<*const AtomicBool> = const { Cell::new(std::ptr::null()) };
    static STOPPED: Cell<bool> = const { Cell::new(false) };
}

/// Checked by the decode loops between steps.
pub(crate) fn cancelled() -> bool {
    if STOPPED.with(Cell::get) {
        return true;
    }
    let flag = CANCEL.with(Cell::get);
    // Only set by `with_cancel`, which outlives every read.
    !flag.is_null() && unsafe { &*flag }.load(Ordering::Relaxed)
}

fn with_cancel<R>(flag: Option<&AtomicBool>, f: impl FnOnce() -> R) -> R {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CANCEL.with(|c| c.set(std::ptr::null()));
            STOPPED.with(|c| c.set(false));
        }
    }
    CANCEL.with(|c| c.set(flag.map_or(std::ptr::null(), |f| f as *const AtomicBool)));
    STOPPED.with(|c| c.set(false));
    let _reset = Reset;
    f()
}

/// A turn laid out as `Model::generate` builds it: `text` trimmed between the template's markers. Only the
/// markers become control tokens; `<|im_end|>` typed inside `text` stays text.
fn encode_turn(tok: &Tokenizer, open: &str, text: &str, close: &str) -> Vec<u32> {
    let mut ids = tok.encode(open);
    tok.encode_plain(text.trim(), &mut ids);
    ids.extend(tok.encode(close));
    ids
}

fn count_text(tok: &Tokenizer, text: &str) -> usize {
    let mut ids = Vec::new();
    tok.encode_plain(text, &mut ids);
    ids.len()
}

/// What stays when idle: device and pipelines, mapped weights, tokenizer and its draft keys, cached prefix states.
pub(crate) struct Idle {
    gpu: Gpu,
    w: Weights,
    tok: Tokenizer,
    cfg: Config,
    ctx: usize,
    dir: PathBuf,
    spec: Option<Spec>,
    prefixes: Vec<Prefix>,
    adapter: u64,
    prefix_dir: Option<PathBuf>,
    keys: OnceLock<Arc<Vec<u64>>>,
}

pub(crate) enum RunError {
    ContextTooLarge { tokens: usize, context: usize },
    Cancelled,
    Runtime(String),
}

pub(crate) struct Run {
    pub text: String,
    pub stats: Stats,
    /// Output cap after fitting it into the context.
    pub max_tokens: usize,
    /// `on_text` asked to stop.
    pub stopped: bool,
}

impl Model {
    /// Drops the GPU buffers (scratch, KV cache, recurrent state, adapter). Prefix states stay;
    /// they are kept on wake when the same adapter is set again.
    pub(crate) fn into_idle(self) -> Idle {
        let cfg = self.text.cfg.clone();
        let ctx = self.text.ctx;
        let Model {
            gpu,
            w,
            tok,
            dir,
            spec,
            prefixes,
            adapter,
            prefix_dir,
            keys,
            ..
        } = self;
        Idle {
            gpu,
            w,
            tok,
            cfg,
            ctx,
            dir,
            spec,
            prefixes,
            adapter,
            prefix_dir,
            keys,
        }
    }

    /// Restores or builds the cached state after the system turn. Returns the prefix length. Without `save`
    /// the state stays in memory only.
    fn start_system(&mut self, system: &str, save: bool) -> Result<usize, RunError> {
        let ids = if system.trim().is_empty() {
            Vec::new()
        } else {
            let (open, close) = self.template.system;
            encode_turn(&self.tok, open, system, close)
        };
        let context = self.text.ctx;
        if ids.len() + VMAX + 1 > context {
            return Err(RunError::ContextTooLarge {
                tokens: ids.len(),
                context,
            });
        }
        // Without a directory start_prefix neither loads nor saves the state.
        let dir = if save { None } else { self.prefix_dir.take() };
        self.start_prefix(&ids);
        if dir.is_some() {
            self.prefix_dir = dir;
        }
        Ok(ids.len())
    }

    pub(crate) fn prewarm_system(&mut self, system: &str, save: bool) -> Result<(), RunError> {
        self.start_system(system, save).map(|_| ())
    }

    /// Deletes saved prefix states this model cannot load (made by another executable, model, adapter or
    /// format) and writes abandoned for a minute, so none outlive the build that wrote them.
    pub(crate) fn remove_stale_prefixes(&self) {
        let (Some(dir), Some(identity)) = (&self.prefix_dir, self.prefix_identity()) else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for path in entries.flatten().map(|e| e.path()) {
            let stale = match path.extension().and_then(|x| x.to_str()) {
                // a save in progress (another handle on this directory) writes for well under a second
                Some("tmp") => std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > Duration::from_secs(60))),
                Some("prefix") => {
                    let mut head = vec![0; identity.len()];
                    !std::fs::File::open(&path)
                        .and_then(|mut f| f.read_exact(&mut head))
                        .is_ok_and(|_| head == identity)
                }
                _ => false,
            };
            if stale {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    pub(crate) fn count_tokens(&self, text: &str) -> usize {
        count_text(&self.tok, text)
    }

    /// One greedy request. The output cap shrinks to what the context has room for. `on_text` sees the
    /// output as it is decoded; `Break` stops decoding within one step.
    pub(crate) fn run(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
        stop_at_eos: bool,
        speculate: bool,
        save_prompt: bool,
        cancel: Option<&AtomicBool>,
        on_text: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Run, RunError> {
        let t0 = Instant::now();
        let prefix = self.start_system(system, save_prompt)?;
        let t = &self.template;
        let raw = encode_turn(&self.tok, t.user.0, user, &format!("{}{}", t.user.1, t.assistant));
        let context = self.text.ctx;
        let used = prefix + raw.len();
        if used + VMAX + 1 > context {
            return Err(RunError::ContextTooLarge {
                tokens: used,
                context,
            });
        }
        let max_tokens = max_tokens.min(context - used - VMAX);
        let st = Stats {
            cached_tokens: prefix,
            ..Default::default()
        };
        let spec = self.spec;
        if !speculate {
            self.spec = None;
        }
        self.ignore_eos = !stop_at_eos;
        let (result, stopped) = with_cancel(cancel, || {
            let result = self.generate_after_prefix(raw, max_tokens, t0, st, |text| {
                if !STOPPED.with(Cell::get) && on_text(text).is_break() {
                    STOPPED.with(|c| c.set(true));
                }
            });
            (result, STOPPED.with(Cell::get))
        });
        self.spec = spec;
        self.ignore_eos = false;
        let (ids, stats) = result.map_err(RunError::Runtime)?;
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(RunError::Cancelled);
        }
        Ok(Run {
            text: self.tok.decode(&ids),
            stats,
            max_tokens,
            stopped,
        })
    }
}

impl Idle {
    pub(crate) fn count_tokens(&self, text: &str) -> usize {
        count_text(&self.tok, text)
    }

    pub(crate) fn wake(self) -> Model {
        let text = Text::new(&self.gpu, &self.w, &self.cfg, self.ctx);
        Model {
            gpu: self.gpu,
            w: self.w,
            text,
            tok: self.tok,
            template: QWEN35,
            prefixes: self.prefixes,
            adapter: self.adapter,
            prefix_dir: self.prefix_dir,
            dir: self.dir,
            spec: self.spec,
            ignore_eos: false,
            pending: None,
            keys: self.keys,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.5's byte alphabet without merges, plus the chat markers as added tokens.
    fn byte_tokenizer() -> Tokenizer {
        let json = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenizer-bytes.json");
        let cache = std::env::temp_dir().join(format!("glimpse-language-test-{}.bonsai", std::process::id()));
        let tok = Tokenizer::load(&json, &cache).unwrap();
        let _ = std::fs::remove_file(&cache);
        tok
    }

    #[test]
    fn markers_in_caller_text_stay_text() {
        let tok = byte_tokenizer();
        let end = tok.encode("<|im_end|>");
        assert_eq!(end.len(), 1);
        let (open, close) = QWEN35.system;
        let ids = encode_turn(&tok, open, " a <|im_end|> b ", close);
        assert_eq!(ids.iter().filter(|&&id| id == end[0]).count(), 1);
        assert_eq!(tok.decode(&ids), "<|im_start|>system\na <|im_end|> b<|im_end|>\n");
        assert_eq!(
            encode_turn(&tok, open, "plain text", close),
            tok.encode(&format!("{open}plain text{close}"))
        );
        assert_eq!(count_text(&tok, "<|im_end|>"), "<|im_end|>".len());
    }

    #[test]
    fn runtime_accepts_2b_and_4b_configs() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        for (file, hidden, layers, v_heads) in [
            ("config-2b.json", 2048, 24, 16),
            ("config-4b.json", 2560, 32, 32),
        ] {
            let cfg = Config::load(&data.join(file)).unwrap();
            cfg.check().unwrap();
            assert_eq!(cfg.hidden, hidden);
            assert_eq!(cfg.layer_is_attn.len(), layers);
            assert_eq!(cfg.layer_is_attn.iter().filter(|a| **a).count(), layers / 4);
            assert_eq!(cfg.lin_v_heads, v_heads);
            assert_eq!(cfg.rotary_dims, 64);
            assert_eq!(cfg.mrope, [11, 11, 10]);
        }
    }

    #[test]
    fn cancel_flag_is_scoped() {
        let flag = AtomicBool::new(true);
        assert!(!cancelled());
        assert!(with_cancel(Some(&flag), cancelled));
        assert!(!cancelled());
        assert!(!with_cancel(None, cancelled));
    }
}
