//! PrismML's Ternary Bonsai 2 (2B and 4B, Qwen3.5 hybrid with ternary weights), single turn: a system
//! prompt whose state is cached, plus user text, decoded greedily with prompt-lookup speculation and an
//! optional LoRA adapter.
//!
//! Runs on Apple Silicon (macOS 14+) through our own Metal runtime in `runtime/`, and on Windows through
//! libllama and ggml in `ggml/` (Vulkan GPU). Everywhere else `availability` reports `Unsupported` and
//! `Bonsai::load` fails.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[cfg(bonsai_metal)]
mod metal;
#[cfg(bonsai_metal)]
use metal as backend;
#[cfg(bonsai_metal)]
#[allow(clippy::all, dead_code, unsafe_op_in_unsafe_fn)]
#[rustfmt::skip]
mod runtime;

#[cfg(bonsai_ggml)]
mod ggml;
#[cfg(bonsai_ggml)]
use ggml as backend;
#[cfg(bonsai_ggml)]
#[path = "runtime/tokenizer.rs"]
#[allow(clippy::all, dead_code)]
#[rustfmt::skip]
mod tokenizer;

/// Files a model directory must contain: PrismML's MLX 2-bit weights on Metal, its Q2_0 GGUF on Windows,
/// plus config.json and tokenizer.json from the MLX pack. The runtime also writes `tokenizer.bonsai` (a
/// parsed tokenizer cache, 6.8 MB) next to them when the directory is writable.
#[cfg(not(bonsai_ggml))]
pub const MODEL_FILES: [&str; 3] = ["model.safetensors", "config.json", "tokenizer.json"];
#[cfg(bonsai_ggml)]
pub const MODEL_FILES: [&str; 3] = ["model.gguf", "config.json", "tokenizer.json"];

/// Positions in the KV cache: system prompt, user turn and output together.
pub const CONTEXT: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    B2,
    B4,
}

impl ModelSize {
    pub const fn label(self) -> &'static str {
        match self {
            Self::B2 => "2B",
            Self::B4 => "4B",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub size: ModelSize,
    pub hidden_size: usize,
    pub layers: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    Available,
    /// Wrong OS, device (no Vulkan GPU on Windows), or build.
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BonsaiErrorKind {
    Unsupported,
    /// Missing or unreadable model files, unsupported config, bad adapter.
    Load,
    /// The prompt leaves no room for output in the context.
    ContextTooLarge,
    Cancelled,
    /// GPU or runtime failure. The model is reloaded on the next call.
    Runtime,
}

#[derive(Debug, Clone)]
pub struct BonsaiError {
    pub kind: BonsaiErrorKind,
    pub message: String,
}

impl std::fmt::Display for BonsaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BonsaiError {}

impl BonsaiError {
    #[cfg_attr(not(bonsai_backend), allow(dead_code))]
    fn new(kind: BonsaiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerateOptions {
    /// Output cap in tokens. Lowered to what the context has room for.
    pub max_tokens: usize,
    /// Stop at the end-of-turn token. Off only for benchmarks.
    pub stop_at_eos: bool,
    /// Prompt-lookup speculation. On Metal the output is token for token the same either way; on Windows
    /// it can differ where two logits nearly tie. It turns itself off within a request when drafts keep
    /// failing.
    pub speculate: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            stop_at_eos: true,
            speculate: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    /// Stopped below `max_tokens` because the context was full.
    ContextFull,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    /// Tokens prefilled for this request (the user turn).
    pub prompt_tokens: usize,
    /// System prompt tokens restored from the cached state, or prefilled and cached now.
    pub cached_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    /// Request start to first output token, including the system prompt when it was not cached.
    pub first_token: Duration,
    pub decode: Duration,
    pub drafted: usize,
    pub accepted: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// Decoded output, trimmed.
    pub text: String,
    pub stop: StopReason,
    pub stats: Stats,
}

/// Whether this build and device can run Bonsai. Cheap after the first call.
pub fn availability() -> Availability {
    #[cfg(bonsai_backend)]
    {
        backend::availability()
    }
    #[cfg(not(bonsai_backend))]
    {
        Availability::Unsupported
    }
}

/// Size and shape of the model in `model_dir`, from its config.json. Does not touch the GPU. Fails when
/// a file in `MODEL_FILES` is missing.
pub fn describe(model_dir: &Path) -> Result<ModelInfo, BonsaiError> {
    for file in MODEL_FILES {
        if !model_dir.join(file).is_file() {
            return Err(BonsaiError::new(
                BonsaiErrorKind::Load,
                format!("{file} missing in {}", model_dir.display()),
            ));
        }
    }
    let path = model_dir.join("config.json");
    let json = std::fs::read_to_string(&path).map_err(|err| {
        BonsaiError::new(BonsaiErrorKind::Load, format!("{}: {err}", path.display()))
    })?;
    parse_config(&json)
}

/// Reads the size and shape from a config.json.
pub fn parse_config(json: &str) -> Result<ModelInfo, BonsaiError> {
    let load = |message: String| BonsaiError::new(BonsaiErrorKind::Load, message);
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|err| load(format!("config.json: {err}")))?;
    let text = &value["text_config"];
    let field = |key: &str| {
        text[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| load(format!("config.json: text_config.{key} missing")))
    };
    let hidden_size = field("hidden_size")?;
    let layers = field("num_hidden_layers")?;
    let vocab_size = field("vocab_size")?;
    let size = match (hidden_size, layers) {
        (2048, 24) => ModelSize::B2,
        (2560, 32) => ModelSize::B4,
        _ => {
            return Err(load(format!(
                "config.json: unknown model (hidden size {hidden_size}, {layers} layers)"
            )));
        }
    };
    Ok(ModelInfo {
        size,
        hidden_size,
        layers,
        vocab_size,
    })
}

/// A loaded model. Calls are serialized; share it behind an `Arc`.
pub struct Bonsai {
    info: ModelInfo,
    #[cfg(bonsai_backend)]
    inner: backend::Handle,
    #[cfg(not(bonsai_backend))]
    inner: Never,
}

#[cfg(not(bonsai_backend))]
enum Never {}

impl Bonsai {
    /// Maps the weights and builds the GPU state (about 25 ms warm on Metal; about 1.5 s on Windows, which
    /// uploads the weights). `adapter` is a LoRA safetensors file: bonsai/train.py's export, or a PEFT
    /// `adapter_model.safetensors` with its `adapter_config.json` beside it. On Windows the first load in a
    /// process also builds the GPU pipelines: about 0.5 s with the driver's shader cache, about a minute
    /// the first time on a machine or after a driver update.
    pub fn load(model_dir: &Path, adapter: Option<&Path>) -> Result<Bonsai, BonsaiError> {
        let info = describe(model_dir)?;
        #[cfg(bonsai_backend)]
        {
            let inner = backend::Handle::load(model_dir, adapter)?;
            Ok(Bonsai { info, inner })
        }
        #[cfg(not(bonsai_backend))]
        {
            let _ = (info, adapter);
            Err(BonsaiError::new(
                BonsaiErrorKind::Unsupported,
                "Bonsai is not supported in this build",
            ))
        }
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Prefills `system_prompt` and caches its state, so a `generate` with the same prompt only prefills
    /// the user text. The three most recent prompts stay cached, also across `release_idle`, and are saved
    /// in the model directory's `prefix-cache` folder for the next load. Blocking.
    pub fn prewarm(&self, system_prompt: &str) -> Result<(), BonsaiError> {
        #[cfg(bonsai_backend)]
        {
            self.inner.prewarm(system_prompt)
        }
        #[cfg(not(bonsai_backend))]
        {
            let _ = system_prompt;
            match self.inner {}
        }
    }

    /// One greedy request. An empty `system_prompt` means none. Blocking; call from a blocking-capable
    /// thread. Setting `cancel` stops decoding within one step and returns `Cancelled`.
    pub fn generate(
        &self,
        system_prompt: &str,
        user_text: &str,
        options: &GenerateOptions,
        cancel: Option<&AtomicBool>,
    ) -> Result<Generation, BonsaiError> {
        #[cfg(bonsai_backend)]
        {
            self.inner
                .generate(system_prompt, user_text, options, cancel)
        }
        #[cfg(not(bonsai_backend))]
        {
            let _ = (system_prompt, user_text, options, cancel);
            match self.inner {}
        }
    }

    /// Frees the GPU buffers; the cached system prompt states stay. On Metal the weights stay mapped and
    /// the next call rebuilds the buffers (and rereads the adapter) in a few tens of ms. On Windows the
    /// weights leave GPU memory too and the next call reloads them, about 1 s. Waits for a running request.
    pub fn release_idle(&self) {
        #[cfg(bonsai_backend)]
        self.inner.release_idle();
        #[cfg(not(bonsai_backend))]
        match self.inner {}
    }

    /// Whether `release_idle` freed the GPU state and no call has rebuilt it yet.
    pub fn is_idle(&self) -> bool {
        #[cfg(bonsai_backend)]
        {
            self.inner.is_idle()
        }
        #[cfg(not(bonsai_backend))]
        match self.inner {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn parses_2b_config() {
        let info = parse_config(&fixture("config-2b.json")).unwrap();
        assert_eq!(
            info,
            ModelInfo {
                size: ModelSize::B2,
                hidden_size: 2048,
                layers: 24,
                vocab_size: 248320,
            }
        );
    }

    #[test]
    fn parses_4b_config() {
        let info = parse_config(&fixture("config-4b.json")).unwrap();
        assert_eq!(info.size, ModelSize::B4);
        assert_eq!(info.size.label(), "4B");
        assert_eq!((info.hidden_size, info.layers), (2560, 32));
    }

    #[test]
    fn rejects_unknown_or_broken_configs() {
        let other =
            r#"{"text_config": {"hidden_size": 4096, "num_hidden_layers": 36, "vocab_size": 1}}"#;
        assert_eq!(parse_config(other).unwrap_err().kind, BonsaiErrorKind::Load);
        assert!(parse_config(r#"{"text_config": {}}"#).is_err());
        assert!(parse_config("not json").is_err());
    }

    #[test]
    fn describe_requires_all_files() {
        let err = describe(Path::new("/nonexistent")).unwrap_err();
        assert_eq!(err.kind, BonsaiErrorKind::Load);
        assert!(err.message.contains(MODEL_FILES[0]));
    }
}
