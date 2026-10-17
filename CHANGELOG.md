# Changelog

## 0.1.0

### Added

- `apple`: Apple's on-device Foundation Models, moved from Glimpse Speech's `cleanup` module (2.0.6) with the same API. Glimpse Speech no longer has the `cleanup-apple` feature.
- `bonsai`: Ternary Bonsai 2 (2B and 4B) on Apple Silicon through a Metal runtime: cached system prompt, prompt-lookup speculation, LoRA adapters, idle release and cancellation. Output matches PrismML's MLX runtime on the golden prompts.
