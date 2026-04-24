# Candle Architecture

The Candle path is the local inference backend in `nexo-ai`. It covers model loading from local files, Apple Silicon execution, and family-specific pipelines for chat, image, audio, and image generation workloads.

## Scope

Candle currently backs these families:

| Family | Runtime variants |
| --- | --- |
| Whisper | safetensors |
| Flux.2 | safetensors |
| Z-Image | GGUF |
| Qwen-Image | safetensors and GGUF |
| Gemma 4 | safetensors and GGUF |

## Layout

The refactor standardizes Candle families under a family-first structure:

```text
models/<family>/
├── common/
│   └── family-owned shared code
├── candle/
│   └── local inference implementation
└── openai/
	└── optional remote adapter
+```

For Candle specifically:

- `common/` holds code shared by multiple backends inside the family
- `candle/` holds the local runtime implementation
- `models/support/` holds cross-family Candle helpers such as encoders and weight loading

## Local Load Flow

```mermaid
flowchart LR
	Manifest[registry manifest<br/>runtime = Candle]
	Factory[coordinator factory]
	Family[family model wrapper]
	Pipeline[family candle pipeline]
	Files[local model files]
	Ready[loaded model slot]

	Manifest --> Factory
	Factory --> Family
	Family --> Pipeline
	Files --> Pipeline
	Pipeline --> Ready
+```

The coordinator does not load tensors directly. It selects the correct family model, and that model delegates to its family-owned Candle pipeline.

## Family Ownership

### Whisper

Whisper is a straightforward local-only family. Its Candle code lives under `models/whisper/candle` and handles transcription-specific preprocessing and decoding.

### Flux.2

Flux.2 uses local Candle pipelines for image generation. Shared family config and sampling code live under `models/flux2/common`, while the transformer and VAE implementation stay in `models/flux2/candle`.

### Qwen-Image And Z-Image

These families both use Candle locally but with different weight formats. The registry runtime metadata tells the coordinator whether the runtime is `candle-safetensors` or `candle-gguf`; loader logic no longer has to guess from filenames.

### Gemma 4

Gemma 4 is the most important mixed-backend family:

- `common/template.rs` owns prompt formatting
- `candle/safetensors` owns the local safetensors path
- `candle/gguf` owns the local GGUF path
- `openai/` owns the remote OpenAI-compatible adapter

That split keeps the family-specific rules together while still making the backend implementation explicit.

## Shared Candle Support

`models/support` contains two kinds of shared code:

- backend-agnostic prompting helpers such as `ChatTemplate`
- Candle-only helpers such as encoder and weight utilities

This matters because prompting is reused by both local and remote paths, while tensor-loading code is only relevant to Candle builds.

## Why The Runtime Metadata Matters

Before the refactor, some runtime choices were inferred indirectly from strings or file extensions. The registry now records the backend directly, which improves three things:

- CLI output can show the real backend
- the coordinator can dispatch without special cases like `"mlx"` as a family
- mixed families such as Gemma 4 can expose both local and remote backends cleanly

## Design Rule

When adding a new local model family, prefer this sequence:

1. add the family root under `models/<family>`
2. put family-shared code under `common/`
3. put local inference code under `candle/`
4. register the family and backend explicitly in the registry
5. add the `(family, runtime)` factory branch in the coordinator

That keeps the backend contract consistent with the rest of the crate.
