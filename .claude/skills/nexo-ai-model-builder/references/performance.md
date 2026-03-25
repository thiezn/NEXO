# Performance Guide

## Common Metal Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| **BF16 dtype on Metal** | ~10x slower than expected. M1/M2/M3 lack hardware BF16 -- every op is software-emulated | Use F32 (safe) or F16 (if weight range permits). Check with a small inference; NaN = F16 overflow |
| **Unoptimized candle in dev builds** | CPU-side tensor management dominates. Debug builds are 10-50x slower for numeric code | `Cargo.toml` has `[profile.dev.package.candle-*] opt-level = 3` -- verify these entries exist |
| **Excessive `.contiguous()` calls** | Each forces a GPU copy kernel. Compounds across layers (e.g. 5 calls x 26 layers = 130/forward) | Audit attention code. Remove `.contiguous()` after `repeat_kv`/`expand` -- `matmul` handles non-contiguous inputs. Keep it only where required (e.g. KV cache `slice_set`) |
| **`Tensor::cat` in `repeat_kv`** | Allocates + copies instead of creating a view | Use `unsqueeze` + `expand` + `reshape` (zero-copy) |
| **RmsNorm dtype round-trips** | BF16->F32->BF16 per norm layer is expensive when BF16 is software-emulated | Unavoidable if using BF16, but switching to F32 eliminates the round-trip entirely |

## Per-Model Checklist

Run through this after implementing a new model:

1. **Run the perf test** and compare tok/s against expected throughput for the model size
2. **Verify Metal is being used** -- `tracing::info` should show "Metal detected"
3. **Count `.contiguous()` calls** in the forward path -- minimize them
4. **Confirm dtype matches hardware capabilities** -- F32 is always safe on Metal
5. **For multi-component pipelines**, verify components are dropped before loading the next

## Performance Test Requirements

Every model must have a performance test using the appropriate macro:

- **Chat/Tool models**: `perf_test!` -- measures tok/s with a warmup pass to prime Metal shaders
- **Talk models**: `talk_perf_test!` -- measures inference time against a maximum threshold

Set conservative minimum thresholds. The goal is catching severe regressions (wrong dtype, missing GPU offload, accidental weight duplication), not tracking marginal changes. If performance is far below expectations, check the pitfalls table above first.

## Benchmarking Tips

- Always use `--release` for meaningful benchmarks: `cargo test -p nexo-ai --release --test model_inference -- --ignored test_my_model_perf`
- The `perf_test!` macro includes a warmup inference to prime Metal shader compilation. Without it, the first inference is significantly slower.
- Compare tok/s across model sizes within a family to spot anomalies (e.g. a 4B model being slower than expected relative to a 12B).
