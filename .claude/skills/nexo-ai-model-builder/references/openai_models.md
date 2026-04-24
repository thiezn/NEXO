# OpenAI-Compatible Models

## When to Use This Path

Use `ModelRuntime::OpenAi { provider, model_repo }` when `nexo-ai` talks to a managed server that exposes an OpenAI-compatible API instead of loading weights directly in Rust.

Current providers:

- `OpenAiProvider::MlxVlm`
- `OpenAiProvider::MlxAudio`

## Shared Transport Layer

The shared OpenAI-compatible stack lives here:

- `nexo-ai/src/openai/protocol.rs` — wire structs
- `nexo-ai/src/openai/client.rs` — HTTP client
- `nexo-ai/src/openai/model.rs` — generic chat/tool/image/multimodal adapter
- `nexo-ai/src/openai/speech.rs` — listen/talk adapter layer

Provider lifecycle is outside these adapters and lives under `nexo-ai/src/servers/`.

## Two Adapter Paths

### `openai/model.rs`

Use this for families that speak through chat completions or compatible multimodal endpoints.

Current example:

- `nexo-ai/src/models/gemma4/openai/mod.rs`

Typical responsibilities:

- implement or reuse an `OpenAiFamilyAdapter`
- resolve the request model id when the manifest name is not the provider request name
- parse provider-native tool calls and fall back to family-local parsing when needed

### `openai/speech.rs`

Use this for listen/talk families that speak through:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`

Current examples:

- `nexo-ai/src/models/whisper/openai/mod.rs`
- `nexo-ai/src/models/voxtral/openai/mod.rs`

Keep family-specific speech request shaping in the family module. Current Voxtral maps `TalkRequest.voice_description` into the provider's `instruct` field in `nexo-ai/src/models/voxtral/common/mod.rs`.

## Provider Wiring

### MLX VLM

- config: `mlx_vlm_host`, `mlx_vlm_port`, `mlx_vlm_venv_path`
- server handle: `ManagedProviderServers::mlx_vlm()`
- process manager: `nexo-ai/src/servers/mlx_vlm/`

### MLX Audio

- config: `mlx_audio_host`, `mlx_audio_port`, `mlx_audio_venv_path`, `mlx_audio_hf_endpoint`
- server handle: `ManagedProviderServers::mlx_audio()`
- process manager: `nexo-ai/src/servers/mlx_audio/`

`CoordinatorConfig::mlx_audio_hf_endpoint()` defaults to `https://hf-mirror.com`, which is the current mirror requirement for managed MLX Audio downloads.

## Load and Unload Semantics

For managed remote models:

- `load()` should ensure the server is running.
- The provider may still lazy-load the actual model weights on the first inference request.
- `unload()` should pass the request model id back to the provider when the provider supports targeted unloads.

This is why provider-managed tests should prove actual inference, not just `load()`.

## Manifest Guidance

Remote manifests typically look like this:

- `family` set to the family enum
- `runtime` set to `ModelRuntime::OpenAi { provider, model_repo }`
- `categories` set normally
- `files: vec![]` when downloads are fully delegated to the provider server

The manifest still matters because it is the canonical place where `ModelFactory` decides how to instantiate the family.

## Testing

Prefer these test surfaces:

- coordinator end-to-end tests for provider-backed manifests
- provider/server tests when you are changing the server process manager itself

Current examples:

- `nexo-ai/tests/mlx_audio_remote.rs`
- `nexo-ai/tests/mlx_server.rs`

Avoid using the REPL as the primary verification surface when you are changing registration, factory wiring, or adapter behavior.
