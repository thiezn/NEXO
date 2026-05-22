# Naming Conventions

Rust standard naming applied to MRPF with project-specific patterns.

## Getter Prefixes by Cost and Ownership

| Prefix | Cost | Ownership |
|--------|------|-----------|
| `as_` | Free | borrowed → borrowed |
| `to_` | Expensive | borrowed → borrowed, borrowed → owned (non-Copy), owned → owned (Copy) |
| `into_` | Variable | owned → owned (non-Copy) |

```rust
// Free conversion — just a reference cast
fn as_bytes(&self) -> &[u8] { &self.data }

// Expensive — allocates or computes
fn to_string(&self) -> String { format!("{}", self) }

// Consumes self — ownership transfer
fn into_inner(self) -> Vec<u8> { self.data }
```

## Type and Module Naming

- **Crate names:** `snake_case` with `mrpf_` prefix (e.g., `mrpf_engine`, `mrpf_tcp_syn_scanner`)
- **Module names:** `snake_case`, keep focused on a single responsibility
- **Types:** `PascalCase` (structs, enums, traits)
- **Constants:** `SCREAMING_SNAKE_CASE`
- **Functions/methods:** `snake_case`

## Boolean Naming

- Prefix with `is_`, `has_`, `can_`, `should_` for clarity
- Avoid bare boolean parameters — use enums instead:

```rust
// Bad — what does `true` mean?
fn scan(target: &str, verbose: bool) { ... }

// Good — self-documenting
enum Verbosity { Quiet, Verbose }
fn scan(target: &str, verbosity: Verbosity) { ... }
```

## Error Type Naming

- Each crate's error enum is just `Error` (not `MycrateError`)
- Re-exported as `pub use error::Error`
- Result alias: `pub type Result<T = (), E = Error> = std::result::Result<T, E>;`

## Module Organization

```rust
// src/lib.rs — re-export public API
pub mod config;
pub mod client;
pub mod error;

pub use config::Config;
pub use client::Client;
pub use error::{Error, Result};
```

- Use `pub(crate)` for internal APIs not exposed to consumers
- Keep `mod.rs` minimal — just `pub mod` and `pub use` declarations

## MRPF-Specific Patterns

- Scanner crates follow the pattern: `mrpf_{protocol}_scanner` (e.g., `mrpf_tls_scanner`)
- Shared library crates use `mrpf_{purpose}` (e.g., `mrpf_core`, `mrpf_matchers`)
- Infrastructure Lambda crates live under `infrastructure/` with clear binary names
- The `Connection` trait in `mrpf_engine` uses verb methods: `build_request()`, `parse_response()`
