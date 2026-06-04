---
name: rust
description: Rust coding conventions. Use when writing, reviewing, or refactoring any Rust code in the workspace.
---

- Only specify major version in Cargo.toml dependencies. Use workspace dependencies to manage versions across crates.
- Prefer proper refactoring instead of small additive changes to maintain backwards compatibility.

# Quality Gates

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

# Module Organization

- Split crates up into submodules
- Keep mod.rs files minimal, only defining modules and exports. The logic should reside in separate files within the module

# Naming Conventions

- Getter Prefixes by Cost and Ownership

| Prefix  | Cost      | Ownership                                                              | example                                               |
| ------- | --------- | ---------------------------------------------------------------------- | ----------------------------------------------------- |
| `as_`   | Free      | borrowed → borrowed                                                    | fn as_bytes(&self) -> &[u8] { &self.data }            |
| `to_`   | Expensive | borrowed → borrowed, borrowed → owned (non-Copy), owned → owned (Copy) | fn to_string(&self) -> String { format!("{}", self) } |
| `into_` | Variable  | owned → owned (non-Copy)                                               | fn into_inner(self) -> Vec<u8> { self.data }          |

# Boolean Naming

- Prefix with `is_`, `has_`, `can_`, `should_`
- Avoid bare boolean parameters - use enums instead

# Error Handling

Every library crate defines its own `Error` enum and `Result<T>` alias:

```rust
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    // Typed variants for each external error
    Other(String),
}
```

- Use `?` everywhere for propagation via `From` conversions
- Binary crates use `Result<(), Box<dyn std::error::Error>>` at `main`
- Optional `ResultExt::context` trait for adding context without `Box` everywhere

See `best-practices/error-handling.md` for full pattern.

# Documentation

- Use `//!` (inner doc comments) at the top of `lib.rs` or `mod.rs` for module/crate documentation.
- Document functions:

````rust
/// Summary.
///
/// # Arguments
///
/// * `param` - A param
///
/// # Examples
///
/// ```ignore
/// some code
/// ```
pub fn new(param: u16) -> Result { ...}
````

# Tracing

- Use `tracing` for all logging and instrumentation
- Always define structured events with fields instead of embedding variables in messages
- Use appropriate levels (`trace`, `debug`, `info`, `warn`, `error`)

# Collections & Iterators

- Prefer Iterators Over Loops
- Use `collect()` Type Inference

# Anti-Patterns

| Anti-Pattern                             | Better Approach                      |
| ---------------------------------------- | ------------------------------------ |
| Boolean parameters                       | Use enums                            |
| `String` parameters when `&str` suffices | Accept `&str` or `impl Into<String>` |
| Long function bodies (>50 lines)         | Extract to smaller functions         |
| Deep nesting (>3 levels)                 | Use early returns                    |
| Magic numbers                            | Use named constants                  |
| `clone()` to satisfy borrow checker      | Restructure ownership                |
