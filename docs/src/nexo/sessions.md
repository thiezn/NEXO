# Inference Sessions

Sessions track multi-turn transcript history between a client and the run loop. Each session maintains a unique `session_id` that flows from the gateway through to the inference node, enabling KV cache reuse across consecutive requests.

---

## 1. Session Lifecycle

### Creation

Sessions are created explicitly via `session.create` or implicitly when a `run.start` request is sent without a `sessionId`. The gateway stores session metadata in SQLite and returns a UUID v7 session ID.

### Resumption

After a client restart, `session.list` returns all previous sessions. The client can resume by including the `sessionId` in subsequent `run.start` requests. The full transcript history is loaded from the gateway's message store and sent to the node as context.

### Session ID propagation

The gateway forwards `session_id` to the inference node as part of the inference payload. This enables the node to track which session's KV cache is currently in memory and decide whether to reuse or replace it.

```
Client                  Gateway                       Node
  |                        |                            |
  |-- run.start ---------->|                            |
  |  { input, sessionId }  |                            |
  |                        |-- inference -------------->|
  |                        |  { messages, session_id }  |
  |                        |                            | prefix matching
  |                        |                            | (reuse KV cache)
  |                        |<-- response ---------------|
  |<-- run event ----------|                            |
```

---

## 2. KV Cache Prefix Reuse

The most impactful optimization in session handling is **KV cache prefix reuse**. In multi-turn transcript flows, each new request contains the full transcript history. Without caching, the model re-processes all previous tokens from scratch on every request. For a 2000-token transcript, this wastes ~90% of prefill time on tokens already seen.

### How it works

The pipeline tracks two pieces of state per session:

- `current_session_id` -- which session's tokens are currently in the KV cache
- `processed_tokens` -- the token IDs that were processed to build the current KV cache state

When a new inference request arrives:

1. The prompt is tokenized into a token sequence.
2. If the `session_id` matches `current_session_id`, the pipeline computes the **longest common prefix** between the new tokens and `processed_tokens`.
3. If a prefix match exists (prefix_len > 0), the KV cache is **not cleared**. Only the new tokens beyond the prefix are fed through the model.
4. If the session differs or there's no match, the KV cache is cleared and all tokens are processed from scratch.

```
Request 1: [sys][user: hello]              → process all tokens, cache them
Request 2: [sys][user: hello][ai: hi][user: how are you?]
           ^^^^^^^^^^^^^^^^^^^^^^ prefix match (reused from cache)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^ only these are processed
```

### What gets cached

The KV cache stores key and value tensors for every attention layer in the model. For Gemma 4 27B, this is 48 layers of key/value pairs. The cache grows with sequence length -- each additional token adds a small tensor slice per layer.

### Limitations

- **Prefix-only matching**: If tokens diverge at position N, the entire cache is cleared. There is no partial cache trimming -- this keeps the implementation simple and correct.
- **Image analysis always clears**: Multimodal forward passes have a different token structure (image embeddings interleaved with text tokens), so prefix matching does not apply.
- **Single session in memory**: Only one session's KV cache is held in GPU/Metal memory at a time. Switching sessions requires saving to disk and loading the other session's cache (see below).

---

## 3. Disk-Persisted KV Cache

When the node needs to switch between sessions, the current session's KV cache is saved to disk so it can be restored later without reprocessing.

### Storage layout

```
~/.nexo/kv_cache/
  <model-name>/
    <session-id>.safetensors   # KV tensors (key + value per layer)
    <session-id>.json          # Metadata sidecar
```

### Metadata sidecar

Each cached session has a JSON metadata file:

```json
{
  "session_id": "01jx...",
  "model_name": "gemma4-27b",
  "processed_tokens": [1, 234, 567, ...],
  "layer_count": 48,
  "created_at": "1743580800",
  "last_accessed": "1743584400"
}
```

The `processed_tokens` array is preserved so that prefix matching works immediately after restoring a session from disk -- the pipeline knows exactly which tokens are already in the KV cache.

### Session switch flow

```
1. Inference request arrives with session_id = B
2. Current in-memory session is A
3. Save session A's KV cache to disk:
   - Move all tensors to CPU
   - Write as safetensors file + JSON metadata
4. Check disk for session B's cache:
   - If found: load tensors to GPU, restore into model, resume prefix matching
   - If not found: clear KV cache, process all tokens from scratch
5. Run inference for session B
```

Tensors are moved to CPU before serialization and back to the target device (Metal GPU) on load. This avoids GPU memory issues during the save operation.

### Cache expiry

Disk caches are automatically expired to manage storage:

| Setting | Default | Description |
|---------|---------|-------------|
| Max entries per model | 8 | Oldest caches are evicted when exceeded |
| Max age | 1 hour | Caches not accessed within this window are deleted |
| Check interval | 5 minutes | Expiry runs periodically after inference completes |

Expiry is based on `last_accessed` -- every time a cache is loaded from disk, its timestamp is updated. This means actively used sessions survive longer than idle ones.

---

## 4. Two-Layer Cache Design

The KV cache system operates at two layers:

| Layer | Location | Handles | Speed |
|-------|----------|---------|-------|
| In-memory prefix matching | nexo-ai (pipeline) | Consecutive requests in same session | Instant (no I/O) |
| Disk persistence | nexo-node (kv_cache module) | Session switching | Seconds (disk I/O + tensor transfer) |

The **common case** -- multiple turns in the same session -- is handled entirely in-memory with zero disk I/O. Disk persistence only activates when the node is asked to serve a different session than the one currently in memory.

### KvCacheable trait

Disk persistence is abstracted behind a `KvCacheable` trait in nexo-ai, following the same pattern as `as_chat()` / `as_tool()`:

```rust
pub trait KvCacheable {
    fn kv_cache_seq_len(&self) -> usize;
    fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>>;
    fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()>;
    fn clear_kv_cache(&mut self);
    fn processed_tokens(&self) -> &[u32];
    fn current_session_id(&self) -> Option<&str>;
    fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>);
    fn tokenizer(&self) -> &tokenizers::Tokenizer;
}
```

Models that support KV caching implement this trait and expose it via `as_kv_cacheable()` on `ModelInfo`. The nexo-node `SessionCacheManager` uses this trait to save and restore caches without knowing model internals.

---

## 5. Observability

The KV cache system logs key events at `DEBUG` level:

| Event | Log message |
|-------|-------------|
| Cache hit (prefix reuse) | `KV cache hit: reusing {N}/{M} tokens` |
| Session switch | `KV cache: switching session from {A} to {B}` |
| New session (no cache) | `KV cache: new session {id}, clearing cache` |
| Disk save | `KV cache saved to disk for session {id} ({N} layers, {T}ms)` |
| Disk load | `KV cache loaded from disk for session {id} ({N} layers, {T}ms)` |
| Disk miss | `KV cache: no disk cache found for session {id}` |
| Expiry | `Expired {N} old KV cache entries from disk` |

Enable debug logging on the node to see these messages:

```bash
nexo-node connect --log-level debug
```
