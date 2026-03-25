---
name: nexo-ai-model-builder
description: Use when adding a new AI model to the nexo-ai framework. Covers model trait implementation, registry registration, inference pipeline setup with Candle, and HuggingFace model inspection.
---

# Adding a New Model to nexo-ai

## Model Categories

| Category | Trait | Input → Output |
|----------|-------|----------------|
| Chat | `ChatModel` | text → text |
| Tool | `ToolModel` | text + tool specs → structured tool calls |
| Image | `ImageModel` | image + text → text |
| Listen | `ListenModel` | audio → text |
| Talk | `TalkModel` | text → audio |
| Imagine | `ImagineModel` | text → image |

Models serving multiple categories go under `models/multipurpose/`.

## Workflow

### 1. Inspect the HuggingFace repo

Use the bundled inspector script to understand the repo structure before writing any code:

```bash
SCRIPT=".claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py"

# Step 1: See directory structure (always start here)
python3 $SCRIPT tree <repo_id> --pretty

# Step 2: Auto-detect components (transformer, vae, tokenizer, etc.)
python3 $SCRIPT autodetect <repo_id> --pretty

# Step 3: Get exact file sizes and SHA-256 hashes for manifest
python3 $SCRIPT inspect <repo_id> --filter "*.safetensors" --sha256 --pretty

# Step 4: Fetch model configs (architecture, tokenizer, preprocessing)
python3 $SCRIPT config <repo_id> --all --pretty

# Step 5: Generate Rust manifest code (supports globs)
python3 $SCRIPT manifest <repo_id> \
  --files "transformer/*.safetensors" "vae/*.safetensors" "tokenizer/tokenizer.json" \
  --component-enum AiComponent --sha256 --pretty
```

**Token resolution order:** `HF_TOKEN` env var → `hugging_token.txt` at project root → `~/.nexo/hf_token.txt` → huggingface-cli cached token.

**Important:** Always run `tree` first to understand the repo layout. Model repos vary widely — some use subdirectories (`transformer/`, `vae/`, `text_encoder/`), others put everything at the root. The `autodetect` command will classify files into components and suggest Rust `AiComponent` mappings.

### 2. Create the model directory

```
nexo-ai/src/models/<category>/<family>/
├── mod.rs         # Model struct, ModelInfo impl, category trait impl
├── config.rs      # Model-specific config types + variant enum if needed
├── pipeline.rs    # Inference pipeline (load, run, sequential load-use-drop)
├── sampling.rs    # Sampling/scheduling utilities (if applicable)
├── transformer.rs # Core model architecture (if porting from Python)
└── vae.rs         # VAE decoder (if applicable, e.g. image models)
```

### 3. Implement traits and register the model

Read `references/trait_registration.md` for the detailed guide on implementing `ModelInfo` + category traits, defining `AiComponent` variants, adding manifest entries, and wiring into the coordinator factory.

### 4. Build the inference pipeline

Read `references/inference_pipeline.md` for the detailed guide on Candle model loading, tokenizer setup, forward pass implementation, token sampling, device/dtype handling, and Metal considerations.

**Key patterns:**
- **Sequential load-use-drop** for large models: load component → use → `drop()` → load next. This minimizes peak memory (critical for >10GB models on unified memory).
- **Config inference from safetensor headers**: For model families with multiple variants (e.g. Flux Klein-4B vs Dev), read the first safetensor file's JSON header to discover architecture params rather than hardcoding for every variant.
- **Avoid `Clone` on large structs**: Never derive `Clone` on structs holding model weights or GPU tensors — an accidental `.clone()` would silently duplicate gigabytes of data.

### 5. Record statistics

After inference completes, call the appropriate `StatsCollector` method on the coordinator:

```rust
self.stats.record_text_generation(model_name, category, tokens_generated, inference_time_ms);
self.stats.record_transcription(model_name, audio_duration_ms, inference_time_ms);
self.stats.record_synthesis(model_name, samples_generated, sample_rate, inference_time_ms);
self.stats.record_image_generation(model_name, images, steps, total_pixels, inference_time_ms);
```

### 6. Test

Before working on a new model, add a integration test in `tests/model_inference.rs` that loads the model and runs a simple inference pass. 

Mark it with `#[ignore]`, `#[timeout]`, and `#[serial]` to prevent it from running in CI or alongside other tests. Use the integration test to validate you have correctly implemented the full pipeline.

## Shared Modules

Reusable modules live in `nexo-ai/src/models/shared/`. These are generic building blocks used across model families.

### Encoders (`models/shared/encoders/`)

| Module | Purpose | Used by |
|--------|---------|---------|
| `t5.rs` | Tokenize text → `[1, seq_len]` tensor via `encode_text()` | Parler TTS |
| `dac.rs` | Decode DAC audio codes → PCM `Vec<f32>` via `decode_to_pcm()` | Parler TTS |
| `qwen3.rs` | Qwen3 BF16 text encoder for Flux.2 image generation | Flux.2 |

**When to add a shared encoder:** If the same encoder architecture (T5, CLIP, Qwen3, etc.) is used by multiple model families, extract it into `models/shared/encoders/`. Family-specific components stay in `models/<category>/<family>/`.

### Weights (`models/shared/weights.rs`)

`find_safetensor_files(model_dir)` — discovers `model.safetensors` (single) or `model-*.safetensors` (sharded) files in a directory. Use this when model files follow the standard `model*.safetensors` naming convention.

**Note:** Some model families use different naming (e.g. `diffusion_pytorch_model.safetensors` for Flux). In those cases, use a local helper that matches `*.safetensors` instead.

### Download Paths (`download/paths.rs`)

`model_storage_dir(model_name)` — returns `~/.nexo/local_models/<sanitized-name>/` with colon-to-dash sanitization. Use this instead of manually joining `default_models_dir()` with the model name.

## Category Dispatch

Models implement `as_<category>()` overrides on `ModelInfo` to enable dynamic dispatch through `ModelSlot`:

```rust
// In your model's ModelInfo impl:
fn as_talk(&mut self) -> Option<&mut dyn TalkModel> { Some(self) }
```

The coordinator uses these to downcast `Box<dyn ModelInfo>` to specific category trait objects at runtime.

## Constructor Pattern

Model constructors receive `memory_bytes` from the coordinator factory (which already has the manifest), rather than re-querying the registry:

```rust
// In coordinator/load.rs — factory passes memory_bytes:
let memory_bytes = (manifest.manifest.size_gb * 1_000_000_000.0) as u64;
Box::new(MyModel::new(model_name.to_string(), memory_bytes, model_dir))

// In the model struct:
pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self { ... }
```

## Manifest Pattern

`ModelFile` is imported at the module level in `registry/manifest.rs`. Use a local `repo` variable to avoid repeating `hf_repo` strings per file entry:

```rust
fn my_model_manifest() -> AiModelManifest {
    let repo = "org/model-name".to_string();
    AiModelManifest {
        manifest: ModelManifest { ..., files: vec![
            ModelFile { hf_repo: repo.clone(), ... },
            ModelFile { hf_repo: repo, ... },
        ]},
        ...
    }
}
```

Wire into `build_all_manifests()` and update the test assertion count.

## REPL Handler Pattern

When adding a new category's handler to the REPL (`cli/repl.rs`):
- Use `coordinator.default_for(category)` to find the model — don't hardcode model names
- Use `coordinator.config().model_settings(&model_name)` for per-model config overrides
- Derive sensible defaults from the manifest family, not from model-specific types (avoid coupling the REPL to a specific model family's config module)
- Use `coordinator.model_mut(&model_name)` + `as_<category>()` for dispatch

## Key Files

| File | Purpose |
|------|---------|
| `nexo-ai/src/shared/model_traits.rs` | `ModelInfo` + category trait definitions |
| `nexo-ai/src/shared/types.rs` | Request/response types per category |
| `nexo-ai/src/shared/lora_traits.rs` | `LoraCapable<C>` trait for adapter hot-swapping |
| `nexo-ai/src/models/shared/encoders/` | Reusable encoder modules (T5, DAC, Qwen3) |
| `nexo-ai/src/models/shared/weights.rs` | `find_safetensor_files()` for model loading |
| `nexo-ai/src/download/paths.rs` | `model_storage_dir()`, `default_models_dir()` |
| `nexo-ai/src/registry/manifest.rs` | `AiComponent` enum + manifest registry |
| `nexo-ai/src/coordinator/load.rs` | `create_model_slot()` factory |
| `nexo-ai/src/statistics/mod.rs` | `StatsCollector` recording methods |
| `nexo-ai/src/device/` | Metal GPU detection, memory checks |

## Implemented Models

| Model | Family | Category | HF Repo | Size |
|-------|--------|----------|---------|------|
| `parler-mini` | parler | Talk | `parler-tts/parler-tts-mini-v1.1` | 3.5 GB |
| `parler-large` | parler | Talk | `parler-tts/parler-tts-large-v1` | 8.7 GB |
| `whisper-large-v3` | whisper | Listen | `openai/whisper-large-v3` | 2.9 GB |
| `whisper-large-v3-turbo` | whisper | Listen | `openai/whisper-large-v3-turbo` | 1.5 GB |
| `distil-large-v3` | whisper | Listen | `distil-whisper/distil-large-v3` | 1.4 GB |
| `flux-2-klein-4b` | flux | Imagine | `black-forest-labs/FLUX.2-klein-4b` | 22 GB |
| `flux-2-klein-9b` | flux | Imagine | `black-forest-labs/FLUX.2-klein-9b` | 49 GB |
| `flux-2-dev` | flux | Imagine | `black-forest-labs/FLUX.2-dev` | 165 GB |

## After Implementation

- Run /simplify to ensure the codebase stays clean and maintainable.
- Review this SKILL.md and make suggestions on improvements or clarifications for future model builders.
