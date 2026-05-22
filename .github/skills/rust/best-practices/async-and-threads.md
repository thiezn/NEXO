# Async vs Threads

When to use `std::thread` vs `tokio` async in MRPF, and why.

## The Core Split

| Component | Model | Why |
|-----------|-------|-----|
| `mrpf_engine` + scanners | `std::thread` | Raw socket I/O needs precise timing, no async overhead |
| `mrpf_scanner_api` | `tokio` async | WebSocket server, concurrent client handling |
| `mrpf_core` (task execution) | `tokio` async | Coordinates multiple scanners, DB access |
| Services (API, workers) | `tokio` async | Network I/O, AWS SDK requires async |
| Infrastructure (Lambda) | `tokio` async | AWS Lambda runtime is async |

## Why mrpf_engine Uses std::thread

The network engine intentionally avoids async for its core send/receive loop:

- **Deterministic timing** — send thread needs precise rate limiting with token bucket
- **No task switching overhead** — dedicated threads for send/receive avoid scheduler latency
- **Raw socket I/O** — `pnet` datalink operations are blocking by nature
- **Simplicity** — three threads (send, receive, status) with channel-based communication
- **Performance** — no allocations from Future state machines in the hot path

```rust
// Engine thread model
let (tx_send, rx_send) = channel();  // send thread ← commands
let (tx_recv, rx_recv) = channel();  // receive thread → results

std::thread::spawn(move || send_loop(rx_send, ...));
std::thread::spawn(move || recv_loop(tx_recv, ...));
// Status thread runs in the calling thread
```

## When to Use tokio Async

- WebSocket handling (`mrpf_scanner_api`)
- HTTP client calls (`mrpf_api_client`)
- Database queries (`mrpf_core` with sqlx)
- AWS SDK operations (SQS, DynamoDB, Lambda)
- Any I/O-bound work with many concurrent connections

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let result = fetch_data().await?;
    Ok(())
}
```

## Avoid Blocking in Async Code

```rust
// BAD — blocks the tokio runtime
async fn bad() {
    std::thread::sleep(Duration::from_secs(1));
}

// GOOD — async sleep
async fn good() {
    tokio::time::sleep(Duration::from_secs(1)).await;
}

// GOOD — spawn_blocking for CPU-intensive work
async fn compute() -> i32 {
    tokio::task::spawn_blocking(|| expensive_computation()).await.unwrap()
}
```

## Channel Patterns

MRPF uses channels extensively for thread/task communication:

- `std::sync::mpsc` — between engine threads (send → receive coordination)
- `tokio::sync::mpsc` — between async tasks (scanner API → clients)
- `tokio::sync::broadcast` — for fan-out (status updates to multiple listeners)

## Rules

- Prefer `std::thread` for anything touching raw sockets or requiring precise timing
- Prefer `tokio` async for everything else (network I/O, AWS, DB)
- Never mix blocking I/O in an async context without `spawn_blocking`
- Use channels for cross-thread communication — avoid shared mutable state
- Keep the send/receive thread model for any new scanner implementations
