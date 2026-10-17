//! Glimpse-Language additions that need the runtime's private items. Ours, not synced from bonsai-ane.

use super::gpu::Gpu;
use super::model::{Config, Text, VMAX};
use super::tokenizer::Tokenizer;
use super::weights::Weights;
use super::{Model, QWEN35, Spec, Stats, Template};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

thread_local! {
    static CANCEL: Cell<*const AtomicBool> = const { Cell::new(std::ptr::null()) };
}

/// Checked by the decode loops between steps.
pub(crate) fn cancelled() -> bool {
    let flag = CANCEL.with(Cell::get);
    // Only set by `with_cancel`, which outlives every read.
    !flag.is_null() && unsafe { &*flag }.load(Ordering::Relaxed)
}

fn with_cancel<R>(flag: Option<&AtomicBool>, f: impl FnOnce() -> R) -> R {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CANCEL.with(|c| c.set(std::ptr::null()));
        }
    }
    CANCEL.with(|c| c.set(flag.map_or(std::ptr::null(), |f| f as *const AtomicBool)));
    let _reset = Reset;
    f()
}

/// The system turn, as `Model::generate` builds it.
pub(crate) fn system_turn(t: &Template, system: &str) -> String {
    format!("{}{}{}", t.system.0, system.trim(), t.system.1)
}

/// The user turn plus the assistant header with the empty think block, as `Model::generate` builds it
/// for text input.
pub(crate) fn user_turn(t: &Template, user: &str) -> String {
    format!("{}{}{}{}", t.user.0, user.trim(), t.user.1, t.assistant)
}

/// What stays when idle: device and pipelines, mapped weights, tokenizer.
pub(crate) struct Idle {
    gpu: Gpu,
    w: Weights,
    tok: Tokenizer,
    cfg: Config,
    ctx: usize,
    dir: PathBuf,
    spec: Option<Spec>,
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
}

impl Model {
    /// Drops the GPU buffers (scratch, KV cache, recurrent state, prefix snapshot, vision).
    pub(crate) fn into_idle(self) -> Idle {
        let cfg = self.text.cfg.clone();
        let ctx = self.text.ctx;
        let Model {
            gpu,
            w,
            tok,
            dir,
            spec,
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
        }
    }

    /// Restores or builds the cached state after the system turn. Returns the prefix length.
    fn start_system(&mut self, system: &str) -> Result<usize, RunError> {
        let ids = if system.trim().is_empty() {
            Vec::new()
        } else {
            self.tok.encode(&system_turn(&self.template, system))
        };
        let context = self.text.ctx;
        if ids.len() + VMAX + 1 > context {
            return Err(RunError::ContextTooLarge {
                tokens: ids.len(),
                context,
            });
        }
        self.start_prefix(&ids);
        Ok(ids.len())
    }

    pub(crate) fn prewarm_system(&mut self, system: &str) -> Result<(), RunError> {
        self.start_system(system).map(|_| ())
    }

    /// One greedy request. The output cap shrinks to what the context has room for.
    pub(crate) fn run(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
        stop_at_eos: bool,
        speculate: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<Run, RunError> {
        let t0 = Instant::now();
        let prefix = self.start_system(system)?;
        let raw = self.tok.encode(&user_turn(&self.template, user));
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
        let result = with_cancel(cancel, || {
            self.generate_after_prefix(raw, None, max_tokens, t0, st, |_| {})
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
        })
    }
}

impl Idle {
    pub(crate) fn wake(self) -> Model {
        let text = Text::new(&self.gpu, &self.w, &self.cfg, self.ctx);
        Model {
            gpu: self.gpu,
            w: self.w,
            text,
            tok: self.tok,
            template: QWEN35,
            vision: None,
            prefix: None,
            dir: self.dir,
            spec: self.spec,
            ignore_eos: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_turn_matches_qwen35() {
        assert_eq!(
            system_turn(&QWEN35, "  Clean up the text.\n"),
            "<|im_start|>system\nClean up the text.<|im_end|>\n"
        );
    }

    #[test]
    fn user_turn_ends_with_empty_think_block() {
        assert_eq!(
            user_turn(&QWEN35, "um hello there "),
            "<|im_start|>user\num hello there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn full_prompt_matches_golden_layout() {
        let prompt = format!("{}{}", system_turn(&QWEN35, "S"), user_turn(&QWEN35, "U"));
        assert_eq!(
            prompt,
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
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
