---
name: local-inference-builder
description: Use when creating a new local ML inference tool in the myclaw project using Rust, Candle, and the local-inference-helpers shared crate. Triggers when building CLI tools that run ML models locally on Metal/CUDA/CPU.
---

# Building Local Inference Tools with Candle

## Critical: Candle Re-exports

Consumer crates MUST use candle through `local-inference-helpers` re-exports. Never add `candle-core` or `candle-nn` as direct dependencies — this causes type mismatches.

```rust
// CORRECT
use local_inference_helpers::candle_core::{Device, Tensor, DType};
use local_inference_helpers::candle_nn::VarBuilder;

// WRONG — will cause type incompatibility
// use candle_core::{Device, Tensor, DType};
```

Only `candle-transformers` (model architecture definitions) should be added as a direct dependency.

## Cargo.toml Feature Flag Pattern

```toml
[features]
default = ["metal"]
cuda = ["local-inference-helpers/cuda", "candle-transformers/cuda"]
metal = ["local-inference-helpers/metal", "candle-transformers/metal"]

[dependencies]
local-inference-helpers = { path = "../../shared/local-inference-helpers", features = ["download"] }
utl-helpers = { path = "../../shared/utl-helpers", features = ["config"] }
candle-transformers = { version = "0", package = "candle-transformers-mold" }
```

Always mirror `cuda`/`metal` features down to both `local-inference-helpers` and `candle-transformers`.

## Key Modules in local-inference-helpers

| Module | Use for | Key functions |
|---|---|---|
| `device` | Hardware selection | `create_device(on_info)` — auto-selects Metal/CUDA/CPU. Override with `LOCAL_INFERENCE_DEVICE=cpu` env var. Also: `fits_in_memory()`, `preflight_memory_check()` |
| `dtype` | Precision selection | `gpu_dtype(device)` — F32 on Metal/CPU, BF16 on CUDA (BF16 has precision issues on Metal) |
| `noise` | Deterministic RNG | `seeded_randn(seed, shape, device, dtype)` — CPU-generated then moved to device. ALWAYS use this instead of `device.set_seed()` + `Tensor::randn()` |
| `progress` | Progress reporting | `ProgressEvent`, `ProgressReporter` with `stage_start()`, `stage_done()`, `info()`, `step()` |
| `paths` | Model storage | `default_models_dir()` -> `~/.myclaw/local_models/`. Override with `LOCAL_INFERENCE_MODELS_DIR` env var |
| `manifest` | Model metadata | `Component` trait, `ModelManifest`, `ModelFile` — defines what files a model needs |
| `download` | Model fetching | `pull_model(manifest)` with progress bars, SHA-256 verification, HF token resolution |

## Component Trait + Manifest Pattern

Each inference crate defines its own component enum describing the model's parts:

```rust
#[derive(Clone, Debug)]
pub enum MyComponent { MainModel, Tokenizer, Config }

impl local_inference_helpers::manifest::Component for MyComponent {
    fn name(&self) -> &str {
        match self { Self::MainModel => "model", Self::Tokenizer => "tokenizer", Self::Config => "config" }
    }
    fn is_model_specific(&self) -> bool {
        // true = stored per-model, false = shared across model variants
        matches!(self, Self::MainModel)
    }
}
```

Then define static manifests with HF repos, file sizes, and SHA-256 hashes:

```rust
pub fn known_manifests() -> Vec<&'static ModelManifest<MyComponent>> { ... }
pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<MyComponent>> { ... }
```

## InferenceEngine Trait + LoadStrategy

Each model family implements `InferenceEngine`:

```rust
pub trait InferenceEngine: Send {
    fn model_name(&self) -> &str;
    fn load(&mut self) -> anyhow::Result<()>;
    fn generate(&mut self, request: &GenerateRequest) -> anyhow::Result<GenerateResponse>;
}
```

Use a factory function to dispatch by model name string:

```rust
pub fn create_engine(model_name, paths, config, load_strategy) -> Box<dyn InferenceEngine>
```

**LoadStrategy** controls memory usage:
- `Eager` — load all components upfront. Fast inference, high peak VRAM.
- `Sequential` — load each component, use it, drop it before loading the next. Essential for memory-constrained environments.

Sequential load-use-drop pattern example:
```rust
// Load text encoder, encode prompt, drop text encoder
let embeddings = { let encoder = load_text_encoder(&paths)?; encoder.encode(&prompt)? };
// Now load the main model into freed memory
let model = load_model(&paths)?;
```

## CLI Structure

Follow the standard subcommand pattern:

- **`generate`** — run inference with CLI args (model name, input, output path, seed, etc.)
- **`pull`** — download model files: `pull_model(manifest)` with terminal progress bars
- **`list`** — show available models, download status, and file sizes

## Tool-Specific Config

Use `utl-helpers` config module for TOML config at `~/.myclaw/<tool_name>.toml`. Store model paths, defaults, and per-model overrides.

## Checklist for a New Inference Tool

1. Create crate under `tools/` with the feature flag pattern above
2. Define a `Component` enum and implement the `Component` trait
3. Build static model manifests with SHA-256 hashes from HF
4. Define request/response types for your domain
5. Implement `InferenceEngine` for each model family
6. Wire up `create_engine` factory
7. Add CLI with `generate`, `pull`, `list` subcommands
8. Use `seeded_randn` for all noise generation (never `Tensor::randn`)
9. Use `gpu_dtype(device)` for precision selection
10. Test with `LOCAL_INFERENCE_DEVICE=cpu` for CI environments
