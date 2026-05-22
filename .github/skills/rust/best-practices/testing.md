# Testing

Unit tests, integration tests, and test patterns used across the MRPF workspace.

## Unit Tests in Same File

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        assert_eq!(add(1, 2), 3);
    }

    #[test]
    fn test_edge_case() {
        assert!(validate("").is_err());
    }
}
```

## Assert Macros

```rust
assert!(condition);
assert_eq!(left, right);
assert_ne!(left, right);
assert!(result.is_ok());
assert!(result.is_err());
assert_matches!(value, Pattern::Variant { .. });
```

## Test Organization Rules

- Unit tests go in a `#[cfg(test)] mod tests` block in the same file as the code
- Integration tests go in `tests/` directory at the crate root
- Test helpers shared across tests go in a `tests/common/mod.rs` file
- Name tests descriptively: `test_parse_valid_certificate`, not `test1`

## Testing Network Code

Network scanners require raw socket access, making integration tests harder:

- **Unit test the parsing logic** — packet construction, response parsing, protocol state machines
- **Mock the network layer** — test `Connection` trait implementations against crafted byte sequences
- **Use test fixtures** — pre-captured packets/responses stored as byte arrays or files
- **Don't test the network** — avoid tests that require actual network access; those are manual integration tests

```rust
#[test]
fn test_parse_tls_certificate() {
    let raw_cert = include_bytes!("../fixtures/example.der");
    let result = parse_certificate(raw_cert);
    assert!(result.is_ok());
    let cert = result.unwrap();
    assert_eq!(cert.common_name, "example.com");
}
```

## Testing Patterns for Iterators

The Feistel cipher iterators in `mrpf_core` need specific testing:

- Verify every element is produced exactly once (completeness)
- Verify no element is outside the valid range (bounds)
- Verify the order is shuffled (not sequential)

```rust
#[test]
fn test_feistel_iterator_completeness() {
    let range = IpRange::new(start, end);
    let items: HashSet<_> = range.iter().collect();
    assert_eq!(items.len(), expected_count);
}
```

## Running Tests

```bash
# All workspace tests
cargo test

# Single crate
cargo test -p mrpf_engine

# Single test
cargo test -p mrpf_engine test_name

# With output
cargo test -p mrpf_engine -- --nocapture

# Only unit tests (no integration tests)
cargo test --lib
```

## CLI Integration Testing

After modifying scanner crates, validate end-to-end behavior by running the `mrpf` CLI against real targets. This is a local-only validation step (not CI).

### Setup

- Passwordless sudo is configured in `/etc/sudoers.d/mrpf` for the local user
- Use `sudo -n` (non-interactive) — Claude Code has no TTY for password prompts
- Always pass `-i en0` — avoids relying on config lookup when HOME changes under sudo
- Claude Code's shell does not source `.zshrc`, so use the **debug binary path** directly: `sudo -n ./target/debug/mrpf --no-color` (from workspace root)

### When to Run

Run CLI integration tests when changes touch:
- Any scanner crate (`mrpf_dns_resolver`, `mrpf_tcp_syn_scanner`, `mrpf_tls_scanner`, `mrpf_http1_scanner`, `mrpf_whois`)
- The core engine (`mrpf_engine`) — test via each affected scanner subcommand
- The CLI itself (`mrpf_cli`) — test whichever subcommands are affected

### Standard Test Target

Use `www.mortimer.nl` / `136.144.153.226` as the default test target. See each crate's skill file (`skills/rust/crates/<crate>.md`) for specific commands.

### Quick Validation Pattern

```bash
# DNS — fastest, good smoke test
sudo -n ./target/debug/mrpf --no-color dns -i en0 www.mortimer.nl

# TCP SYN — port scan
sudo -n ./target/debug/mrpf --no-color tcpsyn -i en0 -p 80,443 136.144.153.226

# TLS — certificate discovery
sudo -n ./target/debug/mrpf --no-color tls -i en0 --snis www.mortimer.nl --targets 136.144.153.226

# HTTP — templated request
sudo -n ./target/debug/mrpf --no-color http -i en0 -p 443 --snis www.mortimer.nl 136.144.153.226

# WHOIS — domain lookup
sudo -n ./target/debug/mrpf --no-color whois -i en0 mortimer.nl
```
