---
name: rust
description: Rust coding conventions for the project. Use when writing, reviewing, or refactoring any Rust code in the workspace. Covers project-specific patterns beyond generic Rust best practices.
---

# Rust Development

## Project-Specific Rules

- **Rust edition:** 2024 (resolver = "3")
- **Workspace lints:** `clippy::unwrap_used = "warn"`, `clippy::expect_used = "warn"`, `clippy::panic = "warn"`
- **Release profile:** `codegen-units = 1`, `lto = "fat"`, `debug = true`
- **Minimal dependencies:** Prefer `std` over crates. Only add external crates when essential.
- **Platform targets:** ARM Linux + macOS. Use `#[cfg(...)]` for platform-specific code.
- **No backwards compatibility:** Aggressive refactoring accepted.

## Quality Gates

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

Run before every commit.

## Error Handling Pattern

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

## Best Practices Index

| Topic | File |
|-------|------|
| Error handling | `best-practices/error-handling.md` |
| Naming conventions | `best-practices/naming.md` |
| Testing patterns | `best-practices/testing.md` |
| Criterion benchmarks | `best-practices/benchmarks.md` |
| Async vs threads | `best-practices/async-and-threads.md` |
| Memory & performance | `best-practices/memory-and-performance.md` |
| API design patterns | `best-practices/api-design.md` |
| Documentation | `best-practices/documentation.md` |
