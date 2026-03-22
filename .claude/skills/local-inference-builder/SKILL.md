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
candle-transformers = "0"
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0"
anyhow = "1"
tokenizers = "0"
safetensors = "0"
rand = "0.9"
dirs = "6"
# Domain-specific deps below (e.g. hound for audio, image for images)
```

Always mirror `cuda`/`metal` features down to both `local-inference-helpers` and `candle-transformers`.

**Important:** `rand = "0.9"` and `safetensors = "0"` must be pinned to match candle's dependency versions. `tokenizers` is needed for HF tokenizer loading. `dirs` is needed for config path resolution.

## Check candle-transformers First

Before implementing model architectures from scratch, check if `candle-transformers` already includes them. As of v0.9.2, it includes: `parler_tts`, `dac`, `snac`, `whisper`, `t5`, `clip`, `flux`, `llama`, `qwen3`, `mimi`, `encodec`, `metavoice`, `csm`, and many more. Use `candle_transformers::models::<model_name>` directly.

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

Then define static manifests with HF repos, file sizes, and SHA-256 hashes. Use `LazyLock` + `HashMap` index for O(1) lookup:

```rust
use std::collections::HashMap;
use std::sync::LazyLock;

static KNOWN_MANIFESTS: LazyLock<Vec<ModelManifest<MyComponent>>> =
    LazyLock::new(|| vec![model_a_manifest(), model_b_manifest()]);

static MANIFEST_INDEX: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    KNOWN_MANIFESTS.iter().enumerate().map(|(i, m)| (m.name.clone(), i)).collect()
});

pub fn known_manifests() -> &'static [ModelManifest<MyComponent>] { &KNOWN_MANIFESTS }
pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<MyComponent>> {
    MANIFEST_INDEX.get(name).map(|&i| &KNOWN_MANIFESTS[i])
}
```

## InferenceEngine Trait + LoadStrategy

Each model family implements `InferenceEngine`:

```rust
pub trait InferenceEngine: Send + Sync {
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> anyhow::Result<()>;
    fn generate(&mut self, request: &GenerateRequest) -> anyhow::Result<GenerateResponse>;
    fn unload(&mut self) {}
    fn set_on_progress(&mut self, _callback: ProgressCallback) {}
}
```

Use a factory function to dispatch by model family string:

```rust
pub fn create_engine(model_name, paths, config, load_strategy) -> Result<Box<dyn InferenceEngine>>
```

The factory resolves model family from config or manifest, then instantiates the appropriate engine.

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

## Recommended Crate Structure

```
tools/<tool_name>/
├── Cargo.toml
└── src/
    ├── main.rs              # async tokio entry, command dispatch (pull/list/generate)
    ├── lib.rs               # public API re-exports
    ├── cli.rs               # clap CLI with generate/pull/list subcommands
    ├── config.rs            # AppConfig + ModelConfig + ModelPaths
    ├── models.rs            # Domain request/response types
    ├── manifest.rs          # Component enum + model manifests registry
    ├── <domain>.rs          # High-level API (e.g. synthesizer.rs, generator.rs)
    └── inference/
        ├── mod.rs           # InferenceEngine trait + LoadStrategy + Request/Response
        ├── factory.rs       # Engine creation by model family
        └── pipelines/
            ├── mod.rs
            └── <family>/    # One directory per model family
                ├── mod.rs
                └── pipeline.rs
```

Key patterns:
- `lib.rs` re-exports the public API so `main.rs` uses `<crate_name>::synthesize()`
- The high-level API function (synthesizer/generator) takes `&AppConfig` to avoid double-loading
- Each pipeline engine uses `Option<LoadedState>` for clean load/unload lifecycle
- `ModelPaths` struct has `from_downloads()` for post-pull config saving and `resolve()` for config lookup

## CLI Structure

Follow the standard subcommand pattern:

- **`generate`** — run inference with CLI args (model name, input, output path, seed, etc.)
- **`pull`** — download model files: `pull_model(manifest)` with terminal progress bars
- **`list`** — show available models, download status, and file sizes

The `pull` command should save model paths to AppConfig after download via `from_downloads()` + `to_model_config()`.

## Tool-Specific Config

Use `utl-helpers` config module for TOML config at `~/.myclaw/<tool_name>.toml`. Store model paths, defaults, and per-model overrides.

Standard config structs:
- `AppConfig` — global defaults + `HashMap<String, ModelConfig>`
- `ModelConfig` — per-model paths (decoder, tokenizer, config_json, shards) + overrides
- `ModelPaths` — resolved `PathBuf` values for inference engines

## Checklist for a New Inference Tool

1. Check `candle-transformers` for existing model implementations before writing custom ones
2. Create crate under `tools/` with the feature flag pattern above
3. Define a `Component` enum and implement the `Component` trait
4. Build static model manifests with HF repos, file sizes, and SHA-256 hashes
5. Implement `AppConfig`, `ModelConfig`, `ModelPaths` in `config.rs`
6. Define domain request/response types in `models.rs`
7. Implement `InferenceEngine` for each model family in `inference/pipelines/`
8. Wire up `create_engine` factory in `inference/factory.rs`
9. Create high-level API function in a synthesizer/generator module
10. Add CLI with `generate`, `pull`, `list` subcommands
11. Use `seeded_randn` for all noise generation (never `Tensor::randn`)
12. Use `gpu_dtype(device)` for precision selection
13. Test with `LOCAL_INFERENCE_DEVICE=cpu` for CI environments
