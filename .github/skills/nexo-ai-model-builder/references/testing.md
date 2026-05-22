# Testing Models

## Test Targets

- `nexo-ai/tests/model_inference.rs` — local Candle model integration tests. Run this target with `--features candle`.
- `nexo-ai/tests/mlx_audio_remote.rs` — coordinator-based end-to-end tests for MLX Audio Whisper and Voxtral.
- `nexo-ai/tests/mlx_server.rs` — MLX VLM provider/server lifecycle and API tests.
- Family unit tests live next to the implementation files, especially templates and request builders.

Choose the narrowest surface that still proves the change.

## Helpers

Shared helpers live in `nexo-ai/tests/common/mod.rs`:

- `init_tracing()`
- `resolve_model(name)`
- `create_test_png()`

`resolve_model()` is for local manifests with real downloaded files. Do not use it for provider-managed remote manifests whose `files` list is empty.

`load_test_audio()` is currently defined inside the test targets that need it.

## Attribute Order

For sync ignored tests, keep the attribute order as:

```rust
#[test]
#[ignore]
#[serial]
#[timeout(600_000)]
fn test_name() { ... }
```

For async provider tests, use `#[tokio::test(flavor = "multi_thread")]` before `#[ignore]`, `#[serial]`, and `#[timeout(...)]`.

## Macros in `model_inference.rs`

Current local-model macros are:

- `listen_test!`
- `imagine_test!`
- `imagine_file_test!`
- `chat_test!`
- `perf_test!`
- `tool_test!`
- `gguf_chat_test!`
- `gguf_image_test!`
- `gguf_audio_test!`
- `image_test!`

Do not document or rely on macros that are not actually present in the file.

## Remote Provider Pattern

For provider-backed manifests, prefer coordinator-level tests over directly calling the adapter constructors.

Pattern:

1. Build a `Coordinator` with provider-specific config.
2. Call `coordinator.load_model(model_name)`.
3. Get the typed capability through `model_mut(model_name)` and `as_listen()` / `as_talk()`.
4. Assert on the real response.
5. Call `unload_all()` and verify cleanup.

Why this path matters: it exercises manifest lookup, factory dispatch, provider server startup, and the transport adapter together.

Remote speech tests also need a Tokio multi-thread runtime because `nexo-ai/src/openai/model.rs` and `nexo-ai/src/openai/speech.rs` call `Handle::current().block_on(...)` internally.

## Commands

Local Candle model tests:

```bash
cargo test -p nexo-ai --features candle --test model_inference -- --ignored test_<name> --nocapture
```

MLX Audio remote end-to-end tests:

```bash
cargo test -p nexo-ai --test mlx_audio_remote -- --ignored --nocapture
```

MLX VLM server tests:

```bash
cargo test -p nexo-ai --test mlx_server -- --ignored --nocapture
```

## Common Issues

- Local model missing: run `cargo run -p nexo-ai --features cli -- pull <model-name>`.
- Local model incomplete: re-download with `--force`.
- Remote provider venv missing dependencies: install them in the interpreter pointed at by the provider config.
- Provider-managed models may lazily download weights on the first request, so the first ignored test run can take much longer than later runs.
- Remote `load()` should only guarantee server readiness. Do not assume warm weights or preloaded tokenizers unless the provider guarantees it.
