//! The public API over the Metal runtime: one model behind a mutex, idle release, panics turned into
//! errors.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

use super::runtime::Model;
use super::runtime::ext::{Idle, RunError};
use super::{
    Availability, BonsaiError, BonsaiErrorKind, CONTEXT, GenerateOptions, Generation, Stats,
    StopReason,
};

pub fn availability() -> Availability {
    static AVAILABILITY: OnceLock<Availability> = OnceLock::new();
    *AVAILABILITY.get_or_init(|| {
        // The kernels are built for macOS 14 and use simdgroup matrices (Apple7, M1 and later).
        let supported = macos_major() >= 14
            && MTLCreateSystemDefaultDevice()
                .is_some_and(|d| d.supportsFamily(MTLGPUFamily::Apple7));
        if supported {
            Availability::Available
        } else {
            Availability::Unsupported
        }
    })
}

fn macos_major() -> u32 {
    let mut buf = [0u8; 32];
    let mut len = buf.len();
    let ok = unsafe {
        libc::sysctlbyname(
            c"kern.osproductversion".as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } == 0;
    if !ok {
        return 0;
    }
    std::str::from_utf8(&buf[..len.saturating_sub(1)])
        .ok()
        .and_then(|v| v.split('.').next()?.parse().ok())
        .unwrap_or(0)
}

enum State {
    Loaded(Box<Model>),
    Idle(Box<Idle>),
    /// After a runtime failure; the next call loads from disk again.
    Unloaded,
}

pub struct Handle {
    dir: PathBuf,
    adapter: Option<PathBuf>,
    state: Mutex<State>,
}

// Metal objects are thread safe, and the mutex serializes every use of the model.
unsafe impl Send for State {}

impl Handle {
    pub fn load(dir: &Path, adapter: Option<&Path>) -> Result<Handle, BonsaiError> {
        if availability() != Availability::Available {
            return Err(BonsaiError::new(
                BonsaiErrorKind::Unsupported,
                "Bonsai needs an Apple Silicon Mac with macOS 14 or later",
            ));
        }
        let model = load_model(dir, adapter)?;
        Ok(Handle {
            dir: dir.to_path_buf(),
            adapter: adapter.map(Path::to_path_buf),
            state: Mutex::new(State::Loaded(Box::new(model))),
        })
    }

    pub fn prewarm(&self, system: &str) -> Result<(), BonsaiError> {
        self.with_model(|model| model.prewarm_system(system))
    }

    pub fn generate(
        &self,
        system: &str,
        user: &str,
        options: &GenerateOptions,
        cancel: Option<&AtomicBool>,
    ) -> Result<Generation, BonsaiError> {
        let run = self.with_model(|model| {
            model.run(
                system,
                user,
                options.max_tokens,
                options.stop_at_eos,
                options.speculate,
                cancel,
            )
        })?;
        let st = run.stats;
        let stop = if st.gen_tokens < run.max_tokens {
            StopReason::Eos
        } else if run.max_tokens < options.max_tokens {
            StopReason::ContextFull
        } else {
            StopReason::MaxTokens
        };
        Ok(Generation {
            text: run.text.trim().to_string(),
            stop,
            stats: Stats {
                prompt_tokens: st.prompt_tokens,
                cached_tokens: st.cached_tokens,
                generated_tokens: st.gen_tokens,
                prefill: Duration::from_secs_f64(st.prefill_s),
                first_token: Duration::from_secs_f64(st.ttft_s),
                decode: Duration::from_secs_f64(st.decode_s),
                drafted: st.drafted,
                accepted: st.accepted,
            },
        })
    }

    pub fn release_idle(&self) {
        let mut state = self.lock();
        *state = match std::mem::replace(&mut *state, State::Unloaded) {
            State::Loaded(model) => State::Idle(Box::new(model.into_idle())),
            other => other,
        };
    }

    pub fn is_idle(&self) -> bool {
        matches!(*self.lock(), State::Idle(_))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the loaded model, waking or reloading it first. A panic drops the model.
    fn with_model<T>(
        &self,
        f: impl FnOnce(&mut Model) -> Result<T, RunError>,
    ) -> Result<T, BonsaiError> {
        let mut state = self.lock();
        let mut model = match std::mem::replace(&mut *state, State::Unloaded) {
            State::Loaded(model) => model,
            State::Idle(idle) => {
                let adapter = self.adapter.as_deref();
                let woken = guarded(move || {
                    let mut model = idle.wake();
                    if adapter.is_some() {
                        model.set_lora(adapter, 1.0)?;
                    }
                    Ok(model)
                })
                .map_err(|message| BonsaiError::new(BonsaiErrorKind::Load, message))?;
                Box::new(woken)
            }
            State::Unloaded => Box::new(load_model(&self.dir, self.adapter.as_deref())?),
        };
        match catch_unwind(AssertUnwindSafe(|| f(&mut model))) {
            Ok(result) => {
                *state = State::Loaded(model);
                result.map_err(run_error)
            }
            Err(panic) => Err(BonsaiError::new(
                BonsaiErrorKind::Runtime,
                panic_message(panic),
            )),
        }
    }
}

fn load_model(dir: &Path, adapter: Option<&Path>) -> Result<Model, BonsaiError> {
    guarded(|| {
        let mut model = Model::load(dir, CONTEXT)?;
        if adapter.is_some() {
            model.set_lora(adapter, 1.0)?;
        }
        Ok(model)
    })
    .map_err(|message| BonsaiError::new(BonsaiErrorKind::Load, message))
}

/// The runtime asserts on bad model files; turn those panics into errors.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| Err(panic_message(panic)))
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "Bonsai runtime panicked".to_string())
}

fn run_error(err: RunError) -> BonsaiError {
    match err {
        RunError::ContextTooLarge { tokens, context } => BonsaiError::new(
            BonsaiErrorKind::ContextTooLarge,
            format!("prompt is {tokens} tokens, context holds {context} with room for output"),
        ),
        RunError::Cancelled => BonsaiError::new(BonsaiErrorKind::Cancelled, "cancelled"),
        RunError::Runtime(message) => BonsaiError::new(BonsaiErrorKind::Runtime, message),
    }
}
