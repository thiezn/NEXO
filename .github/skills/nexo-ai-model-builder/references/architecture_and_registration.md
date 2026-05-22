# Architecture and Registration

## Mental Model

There are three layers to keep straight:

1. `nexo-ai/src/registry/manifest.rs` describes what models exist.
2. `nexo-ai/src/coordinator/factory.rs` decides how a manifest becomes a concrete `Box<dyn ModelInfo>`.
3. `nexo-ai/src/models/<family>/` implements the behavior.

`Coordinator::load_model()` is the end-to-end surface that ties them together, so test that path when you want confidence in the integration.

## Core Enums

The manifest layer currently pivots on these enums:

- `ModelFamily`
- `ModelRuntime`
- `CandleBackend`
- `OpenAiProvider`

Typical runtime split:

- `ModelRuntime::Candle(CandleBackend::Safetensors)`
- `ModelRuntime::Candle(CandleBackend::Gguf)`
- `ModelRuntime::OpenAi { provider, model_repo }`

For provider-managed remote models, `model_repo` is the remote repo or request model id passed to the server-side adapter. The local manifest `files` list can be empty when the provider owns downloads.

## Folder Pattern

Current family code is organized under `nexo-ai/src/models/`:

```text
src/models/
  mod.rs
  support/
  <family>/
    mod.rs
    common/
    candle/        # optional
    openai/        # optional
    gguf/          # optional
    safetensors/   # optional
```

Guidance:

- `mod.rs` owns the public family surface and feature gates.
- `common/` holds family-local logic shared across runtimes, such as request shaping or templates.
- `candle/`, `openai/`, `gguf/`, and `safetensors/` are runtime/backend-specific. Follow the established family layout instead of renaming directories just to normalize them.

Current examples:

- `whisper` — local Candle plus remote OpenAI-compatible speech
- `voxtral` — remote OpenAI-compatible speech plus family-local request shaping
- `gemma4` — local Candle backends, common template code, and OpenAI-compatible chat/tool/image support

## Registration Checklist

When adding or changing a model:

1. Add or extend the family module under `src/models/`.
2. Export the family from `src/models/mod.rs`.
3. Add the manifest function in `src/registry/manifest.rs`.
4. Include it in `build_all_manifests()`.
5. Update manifest tests, including the expected count if it changed.
6. Add the factory match arm in `src/coordinator/factory.rs`.
7. Update any list-view or download assumptions if the runtime is remote.

Remote models are currently treated as effectively downloaded in `nexo-ai/src/registry/models.rs`, which is the right pattern when the provider server owns downloads.

## Coordinator and Provider Notes

`Coordinator::load_model()` does this:

1. Look up the manifest.
2. Run memory preflight.
3. Build a `ModelSlot` through `ModelFactory`.
4. Call `model.load()`.

For local models, `load()` means load weights and become inference-ready.

For managed remote models, `load()` should only ensure the provider server is running. Do not eagerly load every remote model into RAM during coordinator startup.

Provider lifecycle belongs under `nexo-ai/src/servers/`:

- `servers/mod.rs` stores `ManagedProviderServers`
- `servers/mlx_vlm/` manages the MLX VLM process
- `servers/mlx_audio/` manages the MLX Audio process

If a new provider needs different host/port/venv/env settings, extend `CoordinatorConfig`, the provider server handle, and `ManagedProviderServers` together.

## Local Candle Guidance

For local models:

- The family root type should implement `ModelInfo` and the relevant category traits.
- Use `common/` for templates and request shaping shared across Candle backends.
- Keep backend-specific loading and inference paths in backend-specific folders.
- Follow the existing family conventions instead of creating a one-off structure.

`nexo-ai/src/models/gemma4/` is the strongest reference for a family that supports multiple categories and multiple backends.
