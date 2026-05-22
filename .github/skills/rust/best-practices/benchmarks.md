# Benchmarks

Criterion benchmarks and profiling workflow for performance-critical MRPF code.

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
cargo bench -p mrpf_cert_parser

# Specific benchmark
cargo bench -p mrpf_cert_parser -- parse_certificate

# With baseline comparison
cargo bench -p mrpf_cert_parser -- --save-baseline before
# ... make changes ...
cargo bench -p mrpf_cert_parser -- --baseline before
```

## Profiling Workflow

```bash
# Flamegraph (requires cargo-flamegraph)
cargo flamegraph --bin mrpf -- [args]

# macOS Instruments
cargo instruments -t "Time Profiler" --bin mrpf -- [args]

# Release build for accurate profiling (debug symbols enabled in workspace)
cargo build --release
```

The workspace `Cargo.toml` has `debug = true` in the release profile specifically to enable flamegraph profiling without performance overhead.

## When to Benchmark

- Touching packet construction/parsing in `mrpf_engine`
- Modifying certificate parsing in `mrpf_cert_parser`
- Changing matcher evaluation in `mrpf_matchers`
- Modifying the Feistel cipher iterators
- Any change to hot paths (send/receive thread loops)

## Rules

- Always use `black_box()` to prevent compiler from optimizing away the benchmark input
- Compare before/after with `--save-baseline` and `--baseline` flags
- Profile before optimizing — don't guess where bottlenecks are
- Run benchmarks in release mode (criterion does this by default)
- Keep benchmark inputs realistic — use actual packet captures or certificate data
