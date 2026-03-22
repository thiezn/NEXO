# API Design

Builder patterns, newtype safety, and ergonomic public APIs for MRPF crates.

## Builder Pattern for Complex Configuration

```rust
#[derive(Default)]
pub struct ScannerConfig {
    host: Option<String>,
    port: Option<u16>,
    timeout: Option<Duration>,
    rate_limit: Option<u32>,
}

impl ScannerConfig {
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn build(self) -> Result<Scanner> {
        Ok(Scanner {
            host: self.host.unwrap_or_else(|| "0.0.0.0".into()),
            port: self.port.ok_or(Error::MissingPort)?,
            timeout: self.timeout.unwrap_or(Duration::from_secs(30)),
            rate_limit: self.rate_limit.unwrap_or(1000),
        })
    }
}
```

Use builders when a struct has more than 3 configuration fields or when some fields have sensible defaults.

## Newtype Pattern for Type Safety

```rust
// Bad — easy to mix up parameters
fn scan(ip: u32, port: u16, timeout: u64) { ... }

// Good — compile-time safety
pub struct Ipv4Addr(u32);
pub struct Port(u16);
pub struct TimeoutMs(u64);

fn scan(ip: Ipv4Addr, port: Port, timeout: TimeoutMs) { ... }
```

MRPF uses newtypes for IP ranges, port ranges, and other domain types to prevent accidental misuse.

## Accept Generic Input, Return Concrete Output

```rust
// Good — accepts anything string-like
impl Config {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

// Good — borrow when you don't need ownership
fn lookup(key: &str) -> Option<&Value> { ... }
```

## Use `#[must_use]` for Important Returns

```rust
#[must_use]
pub fn validate(&self) -> Result<()> { ... }

#[must_use = "scan results should be processed"]
pub fn run_scan(&self) -> ScanResults { ... }
```

## Visibility Rules

```rust
// Public API — exported from the crate
pub fn scan_target(config: &ScanConfig) -> Result<ScanResults> { ... }

// Internal to crate — not visible to consumers
pub(crate) fn build_packet(target: &Target) -> Vec<u8> { ... }

// Private — only visible in this module
fn compute_checksum(data: &[u8]) -> u16 { ... }
```

## Collections & Iterators

### Prefer Iterators Over Loops

```rust
let results: Vec<_> = items.into_iter()
    .filter(|item| item.is_valid())
    .map(|item| item.transform())
    .collect();
```

### Use `collect()` Type Inference

```rust
let vec: Vec<_> = iter.collect();
let map: HashMap<_, _> = iter.collect();
let results: Result<Vec<_>, _> = iter.collect();  // Short-circuits on first error
```

## Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Boolean parameters | Use enums |
| `String` parameters when `&str` suffices | Accept `&str` or `impl Into<String>` |
| Long function bodies (>50 lines) | Extract to smaller functions |
| Deep nesting (>3 levels) | Use early returns |
| Magic numbers | Use named constants |
| `clone()` to satisfy borrow checker | Restructure ownership |
