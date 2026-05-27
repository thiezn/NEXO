# Error Handling

Per-crate error types with `?` propagation. No `Box` in library code — only at binary boundaries.

## Pattern: Per-Crate Error Enum + Result Alias

Every library crate defines its own error module:

```rust
// src/error.rs
use std::{error::Error as StdError, fmt};

pub type Result<T = (), E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Json(e) => write!(f, "JSON error: {e}"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Json(e) => Some(e),
            Error::Other(_) => None,
        }
    }
}

// From conversions so `?` auto-converts
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Io(e) }
}
```

## Rules

- Use `?` everywhere for propagation — only use explicit `match` when actually handling/recovering
- Add `From<ExternalError>` impls for each external error type the crate encounters
- Binary crates use `Result<(), Box<dyn std::error::Error>>` at `main` — only place `Box` is acceptable
- Implement `Display` short and human-readable; rely on `source()` chains for detail
- `Debug` can be derived

## Context Extension (Optional)

When you need more context without `Box` everywhere:

```rust
use std::borrow::Cow;

pub trait ResultExt<T> {
    fn context(self, msg: impl Into<Cow<'static, str>>) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where E: StdError + Send + Sync + 'static {
    fn context(self, msg: impl Into<Cow<'static, str>>) -> Result<T> {
        self.map_err(|e| Error::Context {
            msg: msg.into(),
            source: Box::new(e),
        })
    }
}
```

## Anti-Patterns

| Bad | Good | Why |
|-----|------|-----|
| `.unwrap()` in library code | `.ok_or_else(\|\| Error::Missing("key"))?` | Workspace lint warns on unwrap |
| `Box<dyn Error>` in library signatures | Typed `Error` enum | Zero-cost, pattern-matchable |
| Deep error chains in hot paths | Minimal error types, log only actionable errors | Performance |
| `panic!` for recoverable errors | `Result` + `?` | Only `panic!` for programmer bugs / invariant violations |
