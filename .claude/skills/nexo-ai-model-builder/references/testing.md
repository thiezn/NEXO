# Testing Models

## Integration Tests

All model integration tests live in `tests/model_inference.rs`. They are `#[ignore]` by default since they require downloaded models and hardware.

### Attribute Order

`#[timeout]` from ntest wraps the test function and **must** come after `#[test]`:

```rust
#[test]
#[ignore]
#[serial]
#[timeout(600_000)]
fn test_my_model() { ... }
```

### Test Macros

Use the existing macros as templates. Each takes a test name, model name, and (for some) a model type path:

| Macro | Category | Usage |
|-------|----------|-------|
| `listen_test!` | Listen | `listen_test!(test_name, "model-name")` |
| `talk_test!` | Talk | `talk_test!(test_name, "model-name", timeout_ms)` |
| `imagine_test!` | Imagine | `imagine_test!(test_name, "model-name")` |
| `chat_test!` | Chat | `chat_test!(test_name, "model-name", model::Type)` or with optional `max_tokens` |
| `tool_test!` | Tool | `tool_test!(test_name, "model-name", model::Type)` or with optional `max_tokens` |
| `perf_test!` | Performance | `perf_test!(test_name, "model-name", min_tok_per_sec, model::Type)` |
| `talk_perf_test!` | Talk perf | `talk_perf_test!(test_name, "model-name", max_seconds)` |

**Every model must have a performance test** using `perf_test!` (for chat models) or `talk_perf_test!` (for talk models). Set conservative minimum thresholds that catch severe regressions without being flaky across hardware variations.

### What the Macros Do

Each macro follows the same pattern:

1. Call `resolve_model(name)` to get the model directory and memory estimate
2. Construct the model struct and call `model.load()`
3. Downcast to the category trait (`as_chat()`, `as_tool()`, etc.)
4. Run inference with a standard request
5. Assert non-empty output
6. `model.unload()` and verify

Performance macros add: warmup inference (to prime Metal shaders), measurement of tok/s or realtime factor, and assertion against a minimum threshold.

### Helper Functions

- `resolve_model(name)` -- Looks up the manifest, checks the model is downloaded, verifies all files exist. Panics with a clear box showing the exact download command if anything is missing.
- `load_test_audio()` -- Loads the test WAV file from `datasets/audio/monkeyinmypocket.wav`.
- `create_test_image()` -- Creates a small solid-red 64x64 PNG in memory.
- `init_tracing()` -- Sets up tracing with `with_test_writer()` so output appears in test captures.

## Running Integration Tests

### Download the model first

```bash
cargo run -p nexo-ai --features cli -- pull <model-name>
```

If the model is gated (requires authentication), ask the user to download it.

### Run specific tests

```bash
cargo test -p nexo-ai --test model_inference -- --ignored test_<model_name> 2>&1
```

Always capture stderr (`2>&1`) -- model loading info and perf metrics are printed there.

### Common issues

- **"MODEL NOT DOWNLOADED"** panic: Download the model with `pull` and retry.
- **"MODEL INCOMPLETE"** panic: Re-download with `--force`:
  ```bash
  cargo run -p nexo-ai --features cli -- pull <model-name> --force
  ```
- **Start with the smallest model** in a family to iterate faster. Fix bugs there before testing larger variants.
- **Examine perf test output** -- look for `PERF:` lines in stderr showing tok/s.

## Unit Tests for Templates

Add template-specific unit tests in the `#[cfg(test)]` module of your `template.rs` file. Test at minimum:

- Single user message formatting
- System message handling
- Multi-turn conversation
- Tool call parsing (with and without reasoning)
- End-of-turn markers
- ReasoningMode behavior (if applicable)

See `models/multipurpose/qwen3/template.rs` and `models/multipurpose/gemma3/template.rs` for comprehensive test examples.
