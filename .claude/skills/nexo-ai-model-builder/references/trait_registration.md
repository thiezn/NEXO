# Trait Implementation & Registry Registration

## 1. Implement ModelInfo

Every model must implement the `ModelInfo` trait from `nexo-ai/src/shared/model_traits.rs`:

```rust
use crate::shared::model_traits::ModelInfo;
use crate::shared::types::ModelCategory;
use anyhow::Result;

pub struct MyModel {
    name: String,
    state: Option<LoadedState>,
    memory_estimate: u64,
}

struct LoadedState {
    model: ModelType,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

impl ModelInfo for MyModel {
    fn name(&self) -> &str { &self.name }
    fn family(&self) -> &str { "my-family" }
    fn categories(&self) -> &[ModelCategory] { &[ModelCategory::Chat] }
    fn memory_estimate_bytes(&self) -> u64 { self.memory_estimate }
    fn is_loaded(&self) -> bool { self.state.is_some() }

    fn load(&mut self) -> Result<()> {
        // Load weights, tokenizer, config into self.state
        let device = crate::device::create_device()?;
        let dtype = crate::device::gpu_dtype(&device);
        // ... load model ...
        self.state = Some(LoadedState { model, tokenizer, config, device });
        Ok(())
    }

    fn unload(&mut self) {
        self.state = None; // Drop frees GPU memory
    }
}
```

### Key points:
- Use `Option<LoadedState>` for clean lifecycle (None = unloaded)
- `unload()` just sets state to None — Rust's Drop handles cleanup
- `memory_estimate_bytes()` should return the approximate GPU memory in bytes

## 2. Implement the category trait

Each category trait extends `ModelInfo` with a single method:

```rust
// Chat
impl ChatModel for MyModel {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        let state = self.state.as_mut().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        // ... run inference ...
        Ok(ChatResponse { text, tokens_generated, inference_time_ms })
    }
}

// Tool
impl ToolModel for MyModel {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> { ... }
}

// Image (analysis)
impl ImageModel for MyModel {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse> { ... }
}

// Listen (speech-to-text)
impl ListenModel for MyModel {
    fn transcribe(&mut self, request: &ListenRequest) -> Result<ListenResponse> { ... }
}

// Talk (text-to-speech)
impl TalkModel for MyModel {
    fn synthesize(&mut self, request: &TalkRequest) -> Result<TalkResponse> { ... }
}

// Imagine (text-to-image)
impl ImagineModel for MyModel {
    fn imagine(&mut self, request: &ImagineRequest) -> Result<ImagineResponse> { ... }
}
```

Request/response types are defined in `nexo-ai/src/shared/types.rs`. Every response includes `inference_time_ms`.

## 3. LoRA support (optional)

If the model supports LoRA adapter hot-swapping, implement `LoraCapable<C>`:

```rust
use crate::shared::lora_traits::{LoraAdapter, LoraCapable, ImageLoraCategory};

impl LoraCapable<ImageLoraCategory> for MyModel {
    fn apply_lora(&mut self, adapter: &LoraAdapter, strength: f32) -> Result<()> {
        // Load and apply LoRA weights
        Ok(())
    }
    fn remove_lora(&mut self) -> Result<()> {
        // Remove current adapter
        Ok(())
    }
}
```

Category enums: `ImageLoraCategory` (HeroImage, BackgroundImage, Object, Style), `ToolLoraCategory` (ToolCalling).

## 4. Register in the manifest

In `nexo-ai/src/registry/manifest.rs`:

### Add component variants if needed

The `AiComponent` enum defines what files a model needs:

```rust
pub enum AiComponent {
    Model,
    ModelShard(u8),
    Tokenizer,
    Config,
    Vae,
    TextEncoder,
    ClipEncoder,
    T5Encoder,
    // Add new variants here if needed
}
```

### Add model manifest

In `build_all_manifests()`, add a new manifest entry:

```rust
fn build_all_manifests() -> Vec<AiModelManifest> {
    vec![
        // ... existing manifests ...
        AiModelManifest {
            manifest: ModelManifest {
                name: "my-model-8b".to_string(),
                family: "my-family".to_string(),
                description: "My Model 8B for chat".to_string(),
                size_gb: 4.5,
                files: vec![
                    ModelFile {
                        component: AiComponent::Model,
                        hf_repo: "org/my-model-8b".to_string(),
                        hf_filename: "model.safetensors".to_string(),
                        size_bytes: 4_831_838_208,
                        gated: false,
                        sha256: Some("abc123..."),
                    },
                    ModelFile {
                        component: AiComponent::Tokenizer,
                        hf_repo: "org/my-model-8b".to_string(),
                        hf_filename: "tokenizer.json".to_string(),
                        size_bytes: 11_421_896,
                        gated: false,
                        sha256: None,
                    },
                    // ... more files ...
                ],
            },
            categories: vec![ModelCategory::Chat],
        },
    ]
}
```

Use `hf_downloader.py manifest` to generate the `ModelFile` snippets with exact sizes and SHA-256 hashes.

## 5. Wire into the coordinator factory

In `nexo-ai/src/coordinator/load.rs`, update `create_model_slot()`:

```rust
fn create_model_slot(&self, model_name: &str) -> Result<ModelSlot> {
    let manifest = find_manifest(model_name)
        .ok_or_else(|| anyhow::anyhow!("unknown model: {model_name}"))?;

    let model: Box<dyn ModelInfo> = match manifest.manifest.family.as_str() {
        "my-family" => {
            let paths = resolve_model_paths(model_name)?;
            Box::new(MyModel::new(model_name.to_string(), paths))
        }
        // ... other families ...
        _ => anyhow::bail!("unsupported model family: {}", manifest.manifest.family),
    };

    Ok(ModelSlot::new(model, manifest.categories.clone()))
}
```

## 6. Register the module

In `nexo-ai/src/models/<category>/mod.rs`, add:

```rust
pub mod my_model;
```

And in the parent `nexo-ai/src/models/mod.rs`, ensure the category module is exported.

## Multipurpose models

Models that serve multiple categories (e.g. chat + tool) go under `models/multipurpose/<model-name>/` and implement multiple traits. Register with all applicable categories:

```rust
categories: vec![ModelCategory::Chat, ModelCategory::Tool],
```
