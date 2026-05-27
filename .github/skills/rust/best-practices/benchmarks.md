# Benchmarks

Criterion benchmarks and profiling workflow for performance-critical code.

## Criterion Setup

Add to crate's `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0", features = ["html_reports"] }

[[bench]]
name = "bench_name"
harness = false
```

Benchmark file in `benches/bench_name.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_parse_certificate(c: &mut Criterion) {
    let raw = include_bytes!("../fixtures/example.der");
    c.bench_function("parse_certificate", |b| {
        b.iter(|| parse_certificate(black_box(raw)))
    });
}

criterion_group!(benches, bench_parse_certificate);
criterion_main!(benches);
```

## Running Benchmarks

```bash
# All benchmarks for a crate
cargo bench -p crate-name

# Specific benchmark
cargo bench -p crate-name -- parse_certificate

# With baseline comparison
cargo bench -p crate-name -- --save-baseline before
# ... make changes ...
cargo bench -p crate-name -- --baseline before
```

## Profiling Workflow

```bash
# Flamegraph (requires cargo-flamegraph)
cargo flamegraph --bin my-binary -- [args]

# macOS Instruments
cargo instruments -t "Time Profiler" --bin my-binary -- [args]

# Release build for accurate profiling (debug symbols enabled in workspace)
cargo build --release
```

The workspace `Cargo.toml` has `debug = true` in the release profile specifically to enable flamegraph profiling without performance overhead.

## Rules

- Always use `black_box()` to prevent compiler from optimizing away the benchmark input
- Compare before/after with `--save-baseline` and `--baseline` flags
- Profile before optimizing — don't guess where bottlenecks are
- Run benchmarks in release mode (criterion does this by default)
- Keep benchmark inputs realistic — use actual packet captures or certificate data
