# Nexo-AI

The `nexo-ai` crate provides a unified, trait-based framework for running AI models locally on Apple Silicon. It manages model discovery, lifecycle, memory, and inference across six model categories while exposing a clean interface for both CLI and gateway consumers.

## Model Categories

| Category | Trait | Purpose |
|----------|-------|---------|
| Chat | `ChatModel` | Text-to-text conversation and summarization |
| Tool | `ToolModel` | Structured output for tool calling |
| Image | `ImageModel` | Image analysis with text prompts |
| Listen | `ListenModel` | Speech-to-text transcription |
| Talk | `TalkModel` | Text-to-speech synthesis |
| Imagine | `ImagineModel` | Text-to-image generation |

All model traits extend `ModelInfo`, which provides lifecycle (`load`/`unload`/`is_loaded`), metadata (`name`/`family`/`categories`), and memory estimation.

## Module Structure

```
nexo-ai/src/
├── shared/           Model traits, LoRA traits, request/response types
├── registry/         Model manifests, discovery, download status
├── download/         Generic manifest types, HF download (feature-gated)
├── coordinator/      Model lifecycle, memory management, slot system
├── statistics/       Inference metrics, running aggregates, display
├── config/           nexo-ai.toml loading/saving (AiConfig)
├── models/           Model implementations per category + multipurpose
│   ├── chat/
│   ├── tool/
│   ├── image/
│   ├── listen/
│   ├── talk/
│   ├── imagine/
│   └── multipurpose/
├── device/           Metal GPU detection, memory FFI, preflight checks
├── audio/            Shared audio preprocessing
├── vision/           Shared image preprocessing and resizing
├── cli/              CLI commands and REPL (behind "cli" feature)
└── lib.rs
```

## Model Traits

Defined in `shared/model_traits.rs`. Each category has a dedicated trait:

- **`ModelInfo`** — Base trait: `name()`, `family()`, `categories()`, `memory_estimate_bytes()`, `is_loaded()`, `load()`, `unload()`
- **`ChatModel`** — `fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse>`
- **`ToolModel`** — `fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse>`
- **`ImageModel`** — `fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse>`
- **`ListenModel`** — `fn transcribe(&mut self, request: &ListenRequest) -> Result<ListenResponse>`
- **`TalkModel`** — `fn synthesize(&mut self, request: &TalkRequest) -> Result<TalkResponse>`
- **`ImagineModel`** — `fn imagine(&mut self, request: &ImagineRequest) -> Result<ImagineResponse>`

Request/response types are in `shared/types.rs`. Each response includes `inference_time_ms` for statistics tracking.

## LoRA Support

`shared/lora_traits.rs` defines `LoraCapable<C>` for models that support adapter hot-swapping. Category enums (`ImageLoraCategory`, `ToolLoraCategory`) classify adapters, and `LoraAdapter` holds the weights path, trigger words, and default strength.

## Registry

The `registry/` module handles model discovery:

- **`manifest.rs`** — `AiComponent` enum (Model, Tokenizer, Config, Vae, etc.), `AiModelManifest` linking a generic `ModelManifest<AiComponent>` to its supported categories. `known_manifests()` returns the static list; `find_manifest()` and `manifests_for_category()` provide lookups.
- **`models.rs`** — `ModelEntry` struct and `list_models()` function that checks download status against `~/.nexo/local_models/`.

The generic `Component`/`ModelManifest`/`ModelFile` types in `download/manifest.rs` are reusable across crates.

## Coordinator

The `coordinator/` module manages model lifecycle:

- **`Coordinator`** — Holds a `HashMap<String, ModelSlot>` of loaded models, active defaults per category, config, and a `StatsCollector`.
- **`load.rs`** — `load_model()` with memory preflight checks, timing instrumentation, and stats recording. `load_defaults()` and `load_startup_models()` for batch loading.
- **`unload.rs`** — `unload_model()`, `unload_all()`, and `free_memory(bytes_needed)` which evicts largest models first.

## Statistics

The `statistics/` module tracks inference performance:

- **`metrics.rs`** — `InferenceRecord` with `InferenceDetail` enum (TextGeneration, Transcription, Synthesis, ImageGeneration). Each variant derives category-specific metrics (tok/s, RTF, x realtime, img/s + step/s).
- **`aggregates.rs`** — `RunningStat` (Welford's online algorithm) for memory-efficient running statistics. `ModelStats` for per-(model, category) aggregates.
- **`backend.rs`** — `StatsBackend` trait with `InMemoryBackend` (VecDeque ring buffer + two-level HashMap aggregates). Designed for future persistence backends.
- **`display.rs`** — CLI table formatting for `/stats` output.
- **`StatsCollector`** — Facade with convenience recording methods and a pluggable backend.

## Configuration

`config/mod.rs` defines `AiConfig` stored at `~/.nexo/nexo-ai.toml`:

- `defaults` — Default model name per category (e.g. `chat = "qwen3-8b"`)
- `startup_categories` — Categories to pre-load on startup
- `models` — Per-model overrides (dtype, max tokens, temperature, etc.)

Uses `utl-helpers` config utilities for TOML load/save.

## CLI

Behind the `cli` feature flag. Three subcommands:

- **`nexo-ai pull [model]`** — Download models from HuggingFace with progress bars and SHA-256 verification
- **`nexo-ai list`** — Show supported/downloaded/loaded models
- **`nexo-ai start`** — Load startup models and enter the interactive REPL

### REPL Commands

| Command | Description |
|---------|-------------|
| `/chat <text>` | Chat with the loaded chat model |
| `/tool <text>` | Send a tool-calling request |
| `/talk <text>` | Synthesize speech from text |
| `/listen` | Record and transcribe audio |
| `/imagine <prompt>` | Generate an image from text |
| `/image <path> <prompt>` | Analyze an image with a prompt |
| `/start models <c,c>` | Load models for categories |
| `/config <key> <value>` | Change a config setting |
| `/list` | Show loaded/available models |
| `/stats [model]` | Show inference performance statistics |
| `/help` | Show help |
| `/quit` | Exit |

Text without a `/` prefix is treated as `/chat` input.

## Device & Memory

The `device/` module handles Metal GPU detection and memory management via FFI calls to macOS APIs. `preflight_memory_check()` verifies sufficient memory before loading a model. `memory_status_string()` provides a human-readable summary of available GPU/system memory.

## Adding New Models

Each model implementation lives under `models/<category>/<model-name>/`. A model must:

1. Implement `ModelInfo` + the appropriate category trait(s)
2. Register in `registry/manifest.rs` with an `AiModelManifest`
3. Wire into the `create_model_slot()` factory in `coordinator/load.rs`

Models that serve multiple categories go under `models/multipurpose/`.

## Gateway Integration

`nexo-gateway` imports `nexo-ai` without the `cli` feature, using the coordinator and traits directly to dispatch inference requests received over WebSocket.
