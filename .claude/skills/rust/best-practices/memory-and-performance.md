# Memory & Performance

Hot path optimization, zero-copy patterns, and allocation strategies for MRPF.

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

## Zero-Copy Parsing

MRPF scanners parse raw packets without copying:

```rust
// Parse directly from the receive buffer slice
fn parse_tcp_header(packet: &[u8]) -> Result<TcpHeader<'_>> {
    // Work with references into the original buffer
    let src_port = u16::from_be_bytes([packet[0], packet[1]]);
    let dst_port = u16::from_be_bytes([packet[2], packet[3]]);
    // ...
}
```

- Parse only what you need — don't deserialize full structures if you only need 2 fields
- Use byte slices (`&[u8]`) through the parsing pipeline
- Only allocate (String, Vec) when you need to store results beyond the buffer lifetime

## Hot Path Rules for MRPF

The send and receive threads in `mrpf_engine` are the hottest paths:

- **No allocations** in the packet send loop
- **No logging** in the tight loop (log only on errors or at intervals)
- **Pre-compute constants** — cipher suites, header templates, etc.
- **Minimize branching** — keep the common path branch-free
- **Use fixed-size buffers** — stack-allocated arrays for packet construction

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
cargo flamegraph --bin mrpf -- [args]    # Flamegraph
cargo instruments -t "Time Profiler"     # macOS Instruments
```
