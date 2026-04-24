# NEXO AI

`nexo-ai` is the inference runtime crate for the NEXO workspace. It provides a single trait-based API for local Candle models and OpenAI-compatible remote backends, plus registry, lifecycle, download, and statistics layers around that API.

The current architecture is built around two explicit ideas:

- model family and execution backend are separate concepts
- model code is organized by family, with backend-specific implementations below each family

## What The Crate Owns

- API traits and request/response types in `src/api`
- static model registry and download metadata in `src/registry`
- lifecycle, loading, and memory management in `src/coordinator`
- backend adapters for local Candle execution and OpenAI-compatible remote execution
- CLI and REPL entry points behind the `cli` feature

## Runtime Categories

The registry currently ships models in these categories:

| Category | Trait | Typical families |
| --- | --- | --- |
| Chat | `ChatModel` | Gemma 4 |
| Tool | `ToolModel` | Gemma 4 |
| Image | `ImageModel` | Gemma 4 |
| Listen | `AudioAnalysisModel` | Whisper, Gemma 4 GGUF |
| Imagine | `ImagineModel` | Flux.2, Qwen-Image, Z-Image |

Every concrete model also implements `ModelInfo`, which provides the shared lifecycle surface: `load`, `unload`, `is_loaded`, `name`, `family`, `categories`, and memory estimation.

## Feature Flags

`nexo-ai` is split by runtime feature instead of one monolithic build:

- `candle`: enables local model implementations
- `mlx`: enables the MLX VLM OpenAI-compatible remote path
- `cli`: enables the terminal CLI and REPL

This matters because some model families, such as Gemma 4, span both local and remote backends.

## Module Layout

The crate now follows a family-first layout:

```text
nexo-ai/src/
├── api/                Traits and request/response types
├── audio/              Shared audio preprocessing
├── cli/                CLI commands and REPL
├── config/             Config loading and runtime overrides
├── coordinator/        Model factory, slot lifecycle, memory policy
├── device/             Device and memory checks
├── download/           Generic manifest/download machinery
├── models/
│   ├── support/        Shared prompting and Candle-only helpers
│   ├── flux2/
│   │   ├── common/
│   │   ├── candle/
│   │   └── openai/
│   ├── gemma4/
│   │   ├── common/
│   │   ├── candle/
│   │   └── openai/
│   ├── qwen_image/
│   ├── whisper/
│   └── z_image/
├── openai/             Generic OpenAI protocol, client, and model adapter
├── registry/           Manifest metadata and list-model view
├── servers/            Provider-specific server integrations
├── statistics/         Inference metrics and aggregates
├── vision/             Shared image preprocessing
└── lib.rs
```

## Family And Runtime Metadata

The registry no longer uses backend names as fake model families. Each manifest now carries:

- `ModelFamily`: `Whisper`, `Flux`, `ZImage`, `QwenImage`, `Gemma4`
- `ModelRuntime`: local Candle backends or remote OpenAI-compatible providers

That split lets the coordinator load the same family through different backends without hard-coding backend names into user-facing metadata.

Examples:

- `gemma-4-e2b-it` is `Gemma4 + candle-safetensors`
- `gemma-4-e2b-it-q5` is `Gemma4 + candle-gguf`
- `mlx-gemma-4-e2b-it-8bit` is `Gemma4 + mlx-vlm`

## Current Families

| Family | Local backends | Remote backends |
| --- | --- | --- |
| Whisper | Candle safetensors | none |
| Flux.2 | Candle safetensors | none |
| Z-Image | Candle GGUF | none |
| Qwen-Image | Candle safetensors and GGUF | none |
| Gemma 4 | Candle safetensors and GGUF | MLX VLM via OpenAI-compatible API |

## Coordinator Role

The coordinator owns:

- active model selection per category
- lazy load and unload operations
- memory preflight checks before large loads
- factory dispatch from registry metadata to concrete model implementations
- statistics recording for load and inference activity

Because runtime metadata is explicit, the loader no longer switches on raw manifest strings such as `"mlx"`.

## CLI Notes

The `list` command now shows both family and backend, which makes mixed-family or mixed-runtime installs easier to inspect.

The REPL uses the same coordinator and model traits as library consumers, so CLI behavior stays aligned with gateway and service integration.

## Where To Read Next

- `Architecture` covers the full crate structure and request/load flows
- `Candle Architecture` explains the local backend layout
- `OpenAI Architecture` explains the generic remote adapter and the MLX server integration
