# Naming Conventions

Rust standard naming with project-specific patterns.

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

- Always prefer to split crates up into submodules
- Use `pub(crate)` for internal APIs not exposed to consumers
- Keep `mod.rs` minimal — just `pub mod` and `pub use` declarations
