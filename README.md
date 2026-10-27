# glimpse-language

On-device language models for Rust, on macOS and Windows. One crate, one Cargo feature per model. What Glimpse Speech is to speech, this is to text.

- **Apple Foundation Models**: Apple's on-device model (macOS 26, Apple Silicon) through a small Swift shim.
- **Ternary Bonsai 2** (2B and 4B): PrismML's ternary Qwen3.5 on our own runtime, Metal on macOS 14 (Apple Silicon) and Vulkan on Windows, with a cached system prompt, speculation from the input text and LoRA adapters.

Requests are single turn: a system prompt plus user text in, text out.

## Cargo features

| Feature | Enables |
| --- | --- |
| `apple` | Apple's on-device Foundation Models (Apple Silicon, macOS 26) |
| `bonsai` | Ternary Bonsai 2 on the GPU (Apple Silicon, macOS 14; Windows with a Vulkan 1.3 GPU) |

On a platform a model does not support, its module still builds and reports itself unsupported, so callers need no `cfg` of their own.

## Installation

```toml
[dependencies]
glimpse-language = { git = "https://github.com/glimpse-hq/Glimpse-Language.git", tag = "0.1.0", features = ["apple", "bonsai"] }
```

### Apple

FoundationModels is weak-linked so the app still launches on macOS 14. Cargo does not pass a dependency's link arguments on, so the app's `build.rs` must repeat them:

```rust
println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
```

### Bonsai

The GPU kernels are compiled at build time and embedded, so the app never compiles shaders.

- macOS needs the Metal toolchain. On Xcode 26 it is a separate download: `xcodebuild -downloadComponent MetalToolchain`.
- Windows needs `glslc` from the Vulkan SDK (`VULKAN_SDK`, or on `PATH`). Vulkan itself loads at run time, so a machine without it reports Bonsai unsupported.

## Usage

### Apple

```rust
use glimpse_language::apple::{apple_availability, apple_generate, apple_prewarm, AppleAvailability};
use std::time::Duration;

if apple_availability() == AppleAvailability::Available {
    apple_prewarm(instructions)?;
    let text = apple_generate(instructions, prompt, 0.0, Some(512), Duration::from_secs(60))?;
}
# Ok::<(), glimpse_language::apple::AppleError>(())
```

`AppleError::kind` separates input problems (`ContextTooLarge`, `Refused`, `UnsupportedLanguage`) from the model being unavailable, busy or timed out.

### Bonsai

```rust
use glimpse_language::bonsai::{availability, Availability, Bonsai, GenerateOptions};

if availability() == Availability::Available {
    let model = Bonsai::load(model_dir, Some(adapter_path))?;
    model.prewarm(system_prompt)?;
    let out = model.generate(system_prompt, user_text, &GenerateOptions { max_tokens: 300, ..Default::default() }, None)?;
    println!("{} ({:?}, {:?})", out.text, out.stop, out.stats);
    model.release_idle();
}
# Ok::<(), glimpse_language::bonsai::BonsaiError>(())
```

- `availability()`: Apple Silicon on macOS 14+, or a Vulkan 1.3 GPU on Windows (NVIDIA, AMD RDNA, Intel Arc or Xe). No CPU fallback.
- `Bonsai::load(dir, adapter)`: loads the model and an optional LoRA adapter.
- `prewarm(system_prompt)`: caches the system prompt so `generate` prefills only the user turn. The last three prompts are cached and saved to `prefix-cache/` for the next launch.
- `generate(...)`: greedy decoding until end of turn or `max_tokens`. `cancel` stops it within one step.
- `release_idle()`: frees the GPU buffers and keeps the weights. Drop the `Bonsai` to free everything.

Speculation drafts tokens from the user's own text and verifies them in one pass. On Metal the output is identical to plain greedy decoding.

### Adapters

A LoRA adapter is a safetensors file: Glimpse-Lab's export, or PEFT `adapter_model.safetensors` with `adapter_config.json` beside it. Saved prompt states are keyed by the adapter's contents, so swapping the file is enough.

### Model files

`model.safetensors`, `config.json` and `tokenizer.json` from PrismML's MLX 2-bit pack (vision tensors optional). Keep the directory writable: the runtime caches the parsed tokenizer and prompt states there.

## Bonsai runtime sync

The runtime is developed in Glimpse-Lab (`g1/rust/`). `src/bonsai/runtime/` is a copy, plus `ext.rs` (ours) and `scripts/bonsai-runtime.patch`. Do not edit the copied files. To pull changes:

```bash
scripts/sync-bonsai.sh ../Glimpse-Lab/g1/rust
cargo test --features bonsai
BONSAI_MODEL_DIR=/path/to/2b cargo test --release --features bonsai --test bonsai_golden -- --ignored
```

The golden test checks output against PrismML's MLX runtime. Current copy: Glimpse-Lab `prefix-cache` at 7d738a8.
