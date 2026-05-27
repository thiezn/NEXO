# Memory & Performance

Hot path optimization, zero-copy patterns, and allocation strategies.

## Core Principles

1. **Avoid heap allocations in hot paths** — use stack/arena buffers
2. **Zero-copy parsing** — work with slices into existing buffers
3. **Drop unused data ASAP** — discard bytes after extracting what you need
4. **Profile before optimizing** — use flamegraphs, not intuition

## Allocation Strategies

### Use `Vec::with_capacity` for Known Sizes

```rust
// Bad — multiple reallocations
let mut vec = Vec::new();
for i in 0..1000 { vec.push(i); }

// Good — single allocation
let mut vec = Vec::with_capacity(1000);
for i in 0..1000 { vec.push(i); }
```

### Return Static References When Possible

```rust
// Bad — allocates even if not needed
fn default_host() -> String { String::from("localhost") }

// Good — no allocation
fn default_host() -> &'static str { "localhost" }
```

### Use `Cow` for Flexible Ownership

```rust
use std::borrow::Cow;

fn process(data: Cow<'_, str>) -> Cow<'_, str> {
    if data.contains("bad") {
        Cow::Owned(data.replace("bad", "good"))
    } else {
        data  // No allocation if unchanged
    }
}
```

## Prefer Borrowing Over Cloning

```rust
// Bad — unnecessary clone
fn process(data: String) { ... }
process(my_string.clone());

// Good — borrow when possible
fn process(data: &str) { ... }
process(&my_string);
```

## Iterators Over Loops

```rust
// Bad — manual loop with push
let mut results = Vec::new();
for item in items {
    if item.is_valid() { results.push(item.transform()); }
}

// Good — iterator chain
let results: Vec<_> = items.iter()
    .filter(|item| item.is_valid())
    .map(|item| item.transform())
    .collect();
```

## Profiling Commands

```bash
cargo build --release                    # Debug symbols enabled in workspace profile
cargo flamegraph --bin my-bin -- [args]  # Flamegraph
cargo instruments -t "Time Profiler"     # macOS Instruments
```
