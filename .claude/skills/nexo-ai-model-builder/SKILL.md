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

Use the bundled inspector script to fetch file sizes, configs, and generate Rust manifest code:

```bash
# List files with exact sizes
python3 .claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py inspect <repo_id> --filter "*.safetensors" --sha256 --pretty

# Fetch model configs (architecture, tokenizer, preprocessing)
python3 .claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py config <repo_id> --all --pretty

# Generate Rust ModelFile snippets
python3 .claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py manifest <repo_id> \
  --files model.safetensors tokenizer.json config.json \
  --component-map model=Model tokenizer=Tokenizer config=Config \
  --component-enum AiComponent --sha256 --pretty
```

The script reads HF tokens from `hugging_token.txt` in the project root or `HF_TOKEN` env var.

### 2. Create the model directory

```
nexo-ai/src/models/<category>/<model-name>/
├── mod.rs         # Model struct, ModelInfo impl, category trait impl
├── config.rs      # Model-specific config types (from HF config.json)
├── pipeline.rs    # Inference pipeline (tokenizer, forward pass, sampling)
└── sampling.rs    # Token sampling (if text generation)
```

### 3. Implement traits and register the model

Read `references/trait_registration.md` for the detailed guide on implementing `ModelInfo` + category traits, defining `AiComponent` variants, adding manifest entries, and wiring into the coordinator factory.

### 4. Build the inference pipeline

Read `references/inference_pipeline.md` for the detailed guide on Candle model loading, tokenizer setup, forward pass implementation, token sampling, device/dtype handling, and Metal considerations.

### 5. Record statistics

After inference completes, call the appropriate `StatsCollector` method on the coordinator:

```rust
// In the coordinator or REPL handler, after inference:
self.stats.record_text_generation(model_name, category, tokens_generated, inference_time_ms);
self.stats.record_transcription(model_name, audio_duration_ms, inference_time_ms);
self.stats.record_synthesis(model_name, samples_generated, sample_rate, inference_time_ms);
self.stats.record_image_generation(model_name, images, steps, total_pixels, inference_time_ms);
```

### 6. Test with REPL

```bash
cargo run --package nexo-ai -- start
# In REPL:
/list                    # verify model appears
/start models <category> # load the model
/chat hello              # or appropriate command
/stats                   # verify metrics
```

## Key Files

| File | Purpose |
|------|---------|
| `nexo-ai/src/shared/model_traits.rs` | `ModelInfo` + category trait definitions |
| `nexo-ai/src/shared/types.rs` | Request/response types per category |
| `nexo-ai/src/shared/lora_traits.rs` | `LoraCapable<C>` trait for adapter hot-swapping |
| `nexo-ai/src/registry/manifest.rs` | `AiComponent` enum + manifest registry |
| `nexo-ai/src/coordinator/load.rs` | `create_model_slot()` factory |
| `nexo-ai/src/statistics/mod.rs` | `StatsCollector` recording methods |
| `nexo-ai/src/device/` | Metal GPU detection, memory checks |

## Candle Re-exports

Consumer crates importing candle through `nexo-ai` must use its re-exports to avoid type mismatches. Only `candle-transformers` (model architectures) should be a direct dependency. Check `candle-transformers` for existing model implementations before writing custom ones.

## Cargo.toml Pattern

```toml
[features]
default = ["metal"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]

[dependencies]
candle-core = "0"
candle-nn = "0"
candle-transformers = "0"
tokenizers = "0"
safetensors = "0"
```
