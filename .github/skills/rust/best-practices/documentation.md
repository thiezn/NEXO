# Documentation

Rustdoc conventions and documentation patterns

## Document Public APIs

```rust
/// Creates a new scanner with the given configuration.
///
/// # Arguments
///
/// * `config` - The scanner configuration
///
/// # Errors
///
/// Returns an error if the configuration is invalid or network
/// interface is not available.
///
/// # Examples
///
/// ```ignore
/// let scanner = Scanner::new(Config::default())?;
/// ```
pub fn new(config: Config) -> Result<Self> { ... }
```

## Documentation Sections

| Section | When to Use |
|---------|-------------|
| Top-level `///` | Always — brief description of what it does |
| `# Arguments` | When parameters aren't obvious from types |
| `# Errors` | When function returns `Result` — list error conditions |
| `# Panics` | When function can panic — list panic conditions |
| `# Examples` | For public API functions — use `ignore` if example needs context |
| `# Safety` | Required for `unsafe` functions |

## Module-Level Documentation

```rust
//! # My Module
//!
//! bla.
//!
//! This module implements ..
```

Use `//!` (inner doc comments) at the top of `lib.rs` or `mod.rs` for module/crate documentation.

## When to Document

- **Always:** Public types, traits, functions, methods
- **Always:** `unsafe` code — explain safety invariants
- **Usually:** Non-obvious internal functions
- **Skip:** Trivial getters, obvious boilerplate, test helpers

## When NOT to Document

- Don't add doc comments to code you didn't change (avoid noise in PRs)
- Don't document every private function — focus on complex or non-obvious logic
- Don't write docs that just restate the function name:

```rust
// Bad — adds no value
/// Gets the name.
fn name(&self) -> &str { &self.name }

// Good — only if there's something non-obvious
/// Returns the normalized hostname (lowercase, no trailing dot).
fn hostname(&self) -> &str { &self.hostname }
```

## Architecture

- The `book/` directory contains prose-level architecture docs — rustdoc covers API-level docs
- the book can be built using ./scripts/book.sh
