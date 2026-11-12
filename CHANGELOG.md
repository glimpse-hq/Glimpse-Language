# Changelog

## 0.2.0

### Added

- `bonsai`: `Bonsai::generate_streaming` hands the output to a callback as it is decoded. The callback can stop the run, which then ends with the new `StopReason::Stopped`.
- `bonsai`: `Bonsai::count_tokens` counts a text's tokens with the model's tokenizer.
- `bonsai`: `device()` names the GPU and its driver version (the macOS version on Metal), for telling when either changes.

### Fixed

- `bonsai`: on Intel graphics without cooperative matrices (Iris Xe), loading no longer crashes the process in Intel's shader compiler, output is no longer garbage that never ends, and prefill no longer resets the GPU (`ERROR_DEVICE_LOST`).
- `bonsai`: on Intel graphics, prefill is up to 4 times faster and speculation no longer slows decoding.

## 0.1.1

### Added

- `apple`: `apple_supported_languages()` lists the base language codes the on-device model supports.

## 0.1.0

### Added

- `apple`: Apple's on-device Foundation Models, moved from Glimpse Speech's `cleanup` module (2.0.6) with the same API. Glimpse Speech no longer has the `cleanup-apple` feature.
- `bonsai`: Ternary Bonsai 2 (2B and 4B), Metal on Apple Silicon and Vulkan on Windows: cached system prompt states (also saved for the next launch), prompt-lookup speculation, LoRA adapters, idle release and cancellation. Output matches PrismML's MLX runtime on the golden prompts.
