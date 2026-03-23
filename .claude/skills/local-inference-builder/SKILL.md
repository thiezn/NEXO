---
name: local-inference-builder
description: Use when creating a new local ML inference tool in the nexo project using Rust, Candle, and the local-inference-helpers shared crate. Triggers when building CLI tools that run ML models locally on Metal/CUDA/CPU.
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
| `paths` | Model storage | `default_models_dir()` -> `~/.nexo/local_models/`. Override with `LOCAL_INFERENCE_MODELS_DIR` env var |
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

## InferenceEngine Trait

Each model family implements `InferenceEngine`. The trait accepts `&Device` and `DType` in `load()` so the caller controls device/precision — this avoids duplicate device creation when the high-level API also needs the device for preprocessing (e.g. image preprocessing in VL models).

```rust
pub trait InferenceEngine: Send {
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self, device: &Device, dtype: DType) -> anyhow::Result<()>;
    fn run(&mut self, request: &DomainRequest) -> anyhow::Result<DomainResponse>;
}
```

The trait only requires `Send` (not `Send + Sync`) because candle models with internal caches are typically not `Sync`. Name the run method after the domain action (`describe`, `generate`, `transcribe`, `synthesize`).

Use a factory function to dispatch by model family:

```rust
pub fn create_engine(model_name: String, paths: ModelPaths) -> Box<dyn InferenceEngine> {
    Box::new(FamilyEngine::new(model_name, paths))
}
```

### LoadedState pattern

Engines use `Option<LoadedState>` for clean lifecycle management:

```rust
struct LoadedState {
    model: ModelType,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

pub struct FamilyEngine {
    name: String,
    paths: ModelPaths,
    state: Option<LoadedState>,
}
```

### Sequential load-use-drop (for multi-component pipelines)

When VRAM is limited and the pipeline has separable stages (e.g. text encoder → diffusion model), load each component, use it, and drop it before loading the next:

```rust
let embeddings = { let encoder = load_text_encoder(&paths)?; encoder.encode(&prompt)? };
// Encoder is dropped, freeing VRAM for the main model
let model = load_model(&paths)?;
```

This isn't needed for autoregressive models (VL, LLM) where the whole model must stay loaded during generation.

## Recommended Crate Structure

```
nexo-tools/<tool_name>/
├── Cargo.toml
└── src/
    ├── main.rs              # async tokio entry, command dispatch (pull/list/domain)
    ├── lib.rs               # public API re-exports
    ├── cli.rs               # clap CLI with domain/pull/list subcommands
    ├── config.rs            # AppConfig + ModelConfig + ModelPaths
    ├── models.rs            # Domain request/response types
    ├── manifest.rs          # Component enum + model manifests registry
    ├── <domain>.rs          # High-level API (e.g. describer.rs, synthesizer.rs)
    ├── <preprocess>.rs      # Input preprocessing (optional, e.g. image_preprocess.rs)
    └── inference/
        ├── mod.rs           # InferenceEngine trait + Request/Response types
        ├── factory.rs       # Engine creation by model family
        └── pipelines/
            ├── mod.rs
            └── <family>/    # One directory per model family
                ├── mod.rs
                ├── pipeline.rs
                └── sampling.rs  # Token sampling (greedy + top-p)
```

Key patterns:
- `lib.rs` re-exports the public API so `main.rs` uses `<crate_name>::describe_image()`
- The high-level API function creates the device once and passes it to both preprocessing and engine loading
- Each pipeline engine uses `Option<LoadedState>` for clean load/unload lifecycle
- `ModelPaths` struct has `from_downloads()` for post-pull config saving and `resolve()` for config lookup

## High-Level API Pattern

The domain function (describer, synthesizer, etc.) creates the device and dtype once and shares them:

```rust
pub fn describe_image(config: &DomainConfig, input: &Path, app_config: &AppConfig) -> anyhow::Result<DomainResult> {
    let start = Instant::now();
    let paths = ModelPaths::resolve(&config.model, app_config)?;
    validate_paths(&paths)?;

    let device = create_device(|info| tracing::info!("{info}"))?;
    let dtype = gpu_dtype(&device);

    // Preprocess input using the same device/dtype
    let preprocessed = preprocess(input, &device, dtype)?;

    // Create and load engine with the SAME device
    let mut engine = create_engine(config.model.clone(), paths);
    engine.load(&device, dtype)?;

    let response = engine.run(&request)?;
    Ok(DomainResult { /* ... */ })
}
```

This avoids creating multiple devices (which can cause Metal context issues and wastes memory).

## CLI Structure

Follow the standard subcommand pattern:

- **`generate`** — run inference with CLI args (model name, input, output path, seed, etc.)
- **`pull`** — download model files: `pull_model(manifest)` with terminal progress bars
- **`list`** — show available models, download status, and file sizes

The `pull` command should save model paths to AppConfig after download via `from_downloads()` + `to_model_config()`.

## Tool-Specific Config

Use `utl-helpers` config module for TOML config at `~/.nexo/<tool_name>.toml`. Store model paths, defaults, and per-model overrides.

Standard config structs:
- `AppConfig` — global defaults + `HashMap<String, ModelConfig>`
- `ModelConfig` — per-model paths (decoder, tokenizer, config_json, shards) + overrides
- `ModelPaths` — resolved `PathBuf` values for inference engines

## Vision-Language Model Patterns

For VL models (e.g. Qwen3-VL), additional patterns apply:

### Image Preprocessing

Add `image = "0"` to deps. Create `image_preprocess.rs` with:

1. **Smart resize**: find (H, W) divisible by `patch_size * spatial_merge_size` (e.g. 28 for Qwen3-VL), keeping total pixels within min/max bounds and preserving aspect ratio
2. **Normalize + patchify in a single fused pass**: iterate over patches directly, normalizing pixel values inline — avoid creating an intermediate normalized image buffer
3. **Temporal frame duplication**: for models with `temporal_patch_size > 1`, write the same normalized value into each temporal slot
4. Return `PreprocessedInput { pixel_values: Tensor, grid_thw: Tensor, num_image_tokens: usize }`

Load preprocessor params from `preprocessor_config.json` with sensible defaults as fallback.

### Chat Template Token Assembly (for VL chat models)

Build input_ids as `Vec<i64>` directly (the target dtype for Tensor) to avoid type conversion:

```rust
fn build_input_ids(tokenizer: &Tokenizer, config: &Config, prompt: &str, num_image_tokens: usize) -> Vec<i64> {
    // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    // <|im_start|>user\n<|vision_start|><|image_pad|>...(N)...<|vision_end|>\n{prompt}<|im_end|>\n
    // <|im_start|>assistant\n
}
```

Helper functions should also return `Vec<i64>` / `Option<i64>` to avoid temporary allocations. Use `ids.resize(ids.len() + N, pad_token_id)` for bulk image pad insertion.

### Continuous Spans for Vision Embedding

Vision models need to know where image tokens are in the input sequence. Find contiguous spans of `image_pad` tokens:

```rust
fn find_continuous_spans(input_ids: &[i64], token_id: i64) -> Vec<(usize, usize)>
```

Pass these spans as `continuous_img_pad` to the model's forward method.

### Autoregressive Generation Loop

```rust
// Prefill: full sequence with vision data
let logits = model.forward(&input_ids_tensor, Some(pixel_values), Some(grid_thw), ...)?;
let mut next_token = sample_token(&last_logits, temperature, top_p)?;

let mut generated = Vec::with_capacity(max_tokens);
for _ in 0..max_tokens {
    if is_eos(next_token) { break; }
    generated.push(next_token);
    // Decode step: single token, no vision data, incremented position
    let logits = model.forward(&token_tensor, None, None, ...)?;
    next_token = sample_token(&logits, temperature, top_p)?;
}
```

### Token Sampling

Create `sampling.rs` with greedy (argmax when temp=0) and top-p nucleus sampling. Use `rand::distr::weighted::WeightedIndex` for top-p.

### Sharded Model Loading

For large models split across multiple safetensors files, `ModelPaths` needs an `all_safetensors()` method:

```rust
pub fn all_safetensors(&self) -> Vec<&Path> {
    let mut paths = vec![self.model_file.as_path()];
    paths.extend(self.model_shards.iter().map(|p| p.as_path()));
    paths
}
```

Use with `VarBuilder::from_mmaped_safetensors(&paths, dtype, device)`.

## HuggingFace Model Inspector (`hf_downloader.py`)

Use `hf_downloader.py` at the project root to fetch exact file metadata from HuggingFace repos. This avoids manual curl calls and ensures manifest file sizes and SHA-256 hashes are accurate.

**Token:** Reads from `hugging_token.txt` in project root, falls back to `HF_TOKEN` env var.

### Subcommands

#### `inspect` — List files with exact sizes
```bash
# All files in a repo
python3 hf_downloader.py inspect openai/whisper-large-v3 --pretty

# Filter by pattern, include SHA-256
python3 hf_downloader.py inspect openai/whisper-large-v3 --filter "*.safetensors" --sha256 --pretty
```
Returns: repo_id, gated status, file list with exact size_bytes, sha256 (LFS only), is_lfs.

#### `config` — Fetch config JSON files
```bash
# Single config file (default: config.json)
python3 hf_downloader.py config parler-tts/parler-tts-mini-v1.1 --pretty

# All standard configs (config.json, generation_config.json, preprocessor_config.json, tokenizer_config.json)
python3 hf_downloader.py config openai/whisper-large-v3 --all --pretty
```
Returns: parsed JSON content of each config file. Missing files are `null`.

#### `manifest` — Generate Rust ModelFile snippet
```bash
python3 hf_downloader.py manifest openai/whisper-large-v3 \
  --files model.safetensors tokenizer.json config.json \
  --component-map model=Model tokenizer=Tokenizer config=Config \
  --component-enum WhisperComponent --sha256 --pretty
```
Returns: file metadata + `rust_code` field with copy-pasteable Rust `ModelFile` structs. Sizes use Rust underscore format (`3_087_130_976`).

#### `verify` — Verify existing manifest data against HF
```bash
echo '[{"hf_repo":"openai/whisper-large-v3","hf_filename":"model.safetensors","expected_size_bytes":3087130976}]' > /tmp/check.json
python3 hf_downloader.py verify --manifest-json /tmp/check.json --pretty
```
Returns: pass/fail for each entry with size/sha mismatch details.

### Workflow for Adding a New Model

1. **Inspect** the repo to find relevant files and exact sizes:
   ```bash
   python3 hf_downloader.py inspect <repo_id> --filter "*.safetensors" --sha256 --pretty
   ```
2. **Fetch configs** to understand model architecture:
   ```bash
   python3 hf_downloader.py config <repo_id> --all --pretty
   ```
3. **Generate manifest** Rust code:
   ```bash
   python3 hf_downloader.py manifest <repo_id> --files <file1> <file2> ... \
     --component-map <stem>=<Name> ... --component-enum <EnumName> --sha256 --pretty
   ```
4. Paste the `rust_code` output into `manifest.rs` and adjust as needed.

### Workflow for Updating an Existing Model

1. **Verify** current manifest sizes match HF:
   ```bash
   python3 hf_downloader.py verify --manifest-json /tmp/check.json --pretty
   ```
2. If mismatches are found, re-inspect and update the manifest with correct values.

## Checklist for a New Inference Tool

1. **Use `hf_downloader.py`** to fetch exact file sizes, SHA-256 hashes, and configs from HuggingFace
2. Check `candle-transformers` for existing model implementations before writing custom ones
3. Create crate under `tools/` with the feature flag pattern above
4. Define a `Component` enum and implement the `Component` trait
5. Build static model manifests with HF repos, file sizes, and SHA-256 hashes (use `hf_downloader.py manifest` to generate)
6. Implement `AppConfig`, `ModelConfig`, `ModelPaths` in `config.rs`
7. Define domain request/response types in `models.rs`
8. Create input preprocessing module if needed (image, audio, etc.)
9. Define `InferenceEngine` trait in `inference/mod.rs` with `load(&mut self, device: &Device, dtype: DType)`
10. Implement engine for each model family in `inference/pipelines/`
11. Wire up `create_engine` factory in `inference/factory.rs`
12. Create high-level API function that creates device ONCE and shares it with preprocessing + engine
13. Add CLI with domain/pull/list subcommands
14. Use `seeded_randn` for all noise generation (never `Tensor::randn`)
15. Use `gpu_dtype(device)` for precision selection
16. Build token IDs in target dtype (i64) from the start — avoid temporary Vec<u32> conversions
17. Test with `LOCAL_INFERENCE_DEVICE=cpu` for CI environments
