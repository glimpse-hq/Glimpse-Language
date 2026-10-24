# glimpse-language

On-device language models for Rust, macOS and Windows. One crate, one Cargo feature per model, each run the fastest way we know on that platform. What Glimpse Speech is to speech, this is to text.

- **Apple Foundation Models**: Apple's on-device model (macOS 26, Apple Silicon) through a small Swift shim
- **Ternary Bonsai 2** (2B and 4B): PrismML's ternary Qwen3.5 hybrid on our own runtime, Metal kernels on macOS 14 (Apple Silicon) and Vulkan kernels on Windows, with a cached system prompt, prompt-lookup speculation and LoRA adapters.

Requests are single turn: a system prompt plus user text in, text out.

## Cargo features

| Feature | Enables |
| --- | --- |
| `apple` | `apple`, generation with Apple's on-device Foundation Models (Apple Silicon, macOS 26) |
| `bonsai` | `bonsai`, Ternary Bonsai 2 on the GPU (Apple Silicon, macOS 14; Windows with a Vulkan 1.3 GPU). Compiles its Metal or Vulkan kernels at build time, see below |

On a platform a model does not support, its module still builds and reports itself unsupported, so callers need no `cfg` of their own.

## Installation

```toml
[dependencies]
glimpse-language = { git = "https://github.com/glimpse-hq/Glimpse-Language.git", tag = "0.1.0", features = ["apple", "bonsai"] }
```

### Apple

The shim builds with a macOS 14 deployment target and checks `#available` on every call, so FoundationModels stays weak. Cargo does not pass a dependency's link arguments to the final binary, so the app's own `build.rs` must repeat the weak link and the Swift runtime path:

```rust
println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
```

Check the release binary with `xcrun dyld_info -imports` and launch it on macOS 14 before shipping.

### Metal toolchain (bonsai)

`build.rs` compiles `src/bonsai/runtime/kernels.metal` with `xcrun metal` and `xcrun metallib`, and the runtime embeds the metallib, so the app never compiles shaders. Xcode 26 ships the Metal toolchain as a separate component. If `xcrun -sdk macosx -f metal` finds nothing, the build stops and says so. On CI:

```bash
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx -f metal
```

### Vulkan toolchain (bonsai, Windows)

`build.rs` compiles `src/bonsai/runtime/kernels/*.comp` to SPIR-V with `glslc` from the Vulkan SDK (`VULKAN_SDK`, or `glslc` on `PATH`), and the runtime embeds it, so the app never compiles shaders. Vulkan itself is loaded at run time (`ash`), so a machine without it runs the app and reports Bonsai unsupported.

## Library

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

`apple_availability` is cheap. `apple_prewarm` keeps one session for the next `apple_generate` with the same instructions. Temperature 0 samples greedily. `AppleError::kind` separates input problems (`ContextTooLarge`, `Refused`, `UnsupportedLanguage`) from the model being unavailable, busy or timed out. This is the API that lived in Glimpse Speech's `cleanup` module up to 2.0.6, unchanged.

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

- `availability()`: Apple Silicon (Apple7 GPU family, M1 and later) on macOS 14 or later, or on Windows a Vulkan 1.3 GPU with buffer device addresses and 32-wide subgroups (NVIDIA; AMD RDNA and Intel Arc or Xe through subgroup size control; the first discrete GPU, else the first integrated one), in a build with the feature. There is no CPU fallback: a 50-word cleanup takes over 20 s on a 24-core Xeon. Cached after the first call.
- `describe(dir)` and `parse_config(json)`: size (`ModelSize::B2` or `B4`) and shape from config.json, no GPU.
- `Bonsai::load(dir, adapter)`: on Metal maps the weights without copying and builds the GPU state, about 25 ms warm; the first request after a cold start pays for paging the weights in (about 0.5 s on the 2B). On Vulkan it uploads the weights, 0.4 to 0.6 s for the 2B and 0.8 to 1.0 s for the 4B warm; the first load on a machine (or after a driver update) also builds the GPU pipelines from the embedded SPIR-V, about 2 s more.
- `prewarm(system_prompt)`: prefills the system prompt and keeps its state on the GPU. `generate` with the same prompt restores it and prefills only the user turn. The three most recent prompts stay cached (about 34 MB each on the 2B with Glimpse's cleanup prompt, 90 MB on the 4B) and are saved to `prefix-cache/` in the model directory, so the first request after a restart skips the system prompt too. A saved state is only used by the same executable, GPU, model files, adapter and prompt tokens.
- `generate(system_prompt, user_text, options, cancel)`: greedy decoding until the end-of-turn token or `max_tokens`. Setting `cancel` stops within one step. The prompt is the Qwen3.5 chat template with an empty think block. `max_tokens` is lowered to the room left in the 2048-position context; `StopReason::ContextFull` says so, and a prompt with no room at all fails with `ContextTooLarge`.
- `release_idle()`: frees the GPU buffers (about 240 MB on the 2B) and keeps the weights (mapped on Metal, in GPU memory on Vulkan: 0.54 GB for the 2B, 1.2 GB for the 4B) and the cached prompt states. The next call rebuilds them in a few tens of ms and rereads the adapter. Drop the `Bonsai` to free the weights.
- Calls on one `Bonsai` are serialized by a mutex. A runtime panic becomes a `Runtime` error and the model reloads on the next call.

Speculation drafts up to 7 tokens from the text inside the request's `<transcript>` or `<text>` block (the whole user turn without one): fillers dropped, matched on lowercase words (n-grams of 3 down to 1), with the casing and apostrophes a cleanup adds. Drafts are verified in one pass; the draft length follows recent acceptance, and the request decodes plainly while drafting does not pay. On Metal the output is token for token the plain greedy output; on Vulkan it can differ only where two logits nearly tie.

LoRA adapters are safetensors files: bonsai/train.py's export, or PEFT `adapter_model.safetensors` with `adapter_config.json` beside it. Decode cost is 0 to 2% on Metal and about 7% on Vulkan (fused into the matvec epilogue, and run as extra k steps of the cooperative-matrix matmuls).

On Windows the runtime is the same Rust (model, tokenizer, LoRA, speculation, prefix cache) over `runtime/gpu_vk.rs` and GLSL ports of the Metal kernels, with these differences:
- Matmuls over more than one token (verify, prefill) run on 16x16x16 cooperative matrices: weights are exact in fp16 and each activation is split into fp16 hi and lo parts, so products keep about 22 bits. GPUs without them (or with `BONSAI_NO_COOPMAT=1`) use scalar kernels: same outputs, about twice the prefill time.
- Command streams seen before are replayed from cached command buffers, with the attention offset and split count read from a parameter slot, so a decode or verify step does not re-record about 280 dispatches.
- Speculation uses its own cost table (`gpu::VERIFY_COST_*`): a verify round costs about 2.2 decode steps whether it checks 1 or 7 drafts.
- On the 239-prompt suite (Glimpse's cleanup cases plus Glimpse-Lab's) the greedy output matched the Metal runtime on every prompt for the 2B and the 4B, on an RTX 4000 Ada and an RTX 3080.
- `BONSAI_PROFILE=1` prints per-kernel GPU times (and GB/s for the matmuls) when the model is dropped.

### Expected model files

| Model | Required files |
| --- | --- |
| Ternary Bonsai 2 2B or 4B (macOS and Windows) | `model.safetensors`, `config.json` and `tokenizer.json` from PrismML's MLX 2-bit pack |

The runtime writes a parsed tokenizer cache, `tokenizer.bonsai` (6.8 MB), next to `tokenizer.json`, and saved prompt states in `prefix-cache/`. Keep the model directory writable or the tokenizer is parsed on every load (about 300 ms) and every new process prefills its system prompt.

## Bonsai runtime sync

The runtime (shared Rust, the Metal and Vulkan layers, `kernels.metal` and `kernels/`) is developed in Glimpse-Lab (`g1/rust/`). `src/bonsai/runtime/` holds a copy with `crate::` paths rewritten to `super::`, plus `ext.rs` (ours: template, idle release, cancel, context checks) and `scripts/bonsai-runtime.patch` (adds `mod ext` and the cancel checks in the decode loops). To pull runtime changes:

```bash
scripts/sync-bonsai.sh ../Glimpse-Lab/g1/rust
cargo test --features bonsai
BONSAI_MODEL_DIR=/path/to/2b cargo test --release --features bonsai --test bonsai_golden -- --ignored
BONSAI_MODEL_DIR=/path/to/4b cargo test --release --features bonsai --test bonsai_golden -- --ignored
```

If the patch no longer applies, redo its small changes by hand and regenerate it with `diff -u`. Do not edit the copied files directly. Current copy: Glimpse-Lab `g1/rust/` on branch `vulkan-runtime` (6a42f93), 2026-10-23.

The golden test compares output with PrismML's MLX runtime on the prompts in `tests/golden`, cold, from the cached prompt and without speculation. It runs on Windows too (Vulkan).

Known runtime gaps: weights are never unmapped, so dropping a `Bonsai` leaves the mapping (clean file pages) until exit; input is not NFC normalized before tokenizing; the vision tower is compiled but not exposed.

## Examples

```bash
cargo run --example apple --features apple
cargo run --release --example bonsai --features bonsai -- <model dir> [adapter.safetensors]
cargo run --release --example bonsai_suite --features bonsai -- <model dir> <cases.json> <out.jsonl> [--adapter FILE] [--no-spec] [--repeat N]
```

`bonsai_suite` runs a list of requests and writes the output and timings per request; compare the files from two machines to check parity.

## Acknowledgments

- [Ternary Bonsai 2](https://huggingface.co/prism-ml) by PrismML (Apache 2.0), based on Qwen3.5. Ship its `LICENSE` and `NOTICE.txt` with the model files.
