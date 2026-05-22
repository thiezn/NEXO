# Documentation

Rustdoc conventions and documentation patterns for MRPF crates.

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
//! # Network Engine
//!
//! Core packet crafting and parsing for the MRPF scanner.
//!
//! This module implements the custom network stack using `libpnet`
//! at the datalink layer, with separate send/receive threads.
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

## MRPF-Specific Notes

- Performance-critical sections should have comments explaining trade-offs (e.g., why a constant cipher suite array is used)
- Document the `Connection` trait thoroughly — it's the main extension point for new scanners
- Crate READMEs serve as the entry point for each crate's documentation
- The `book/` directory contains prose-level architecture docs — rustdoc covers API-level docs
