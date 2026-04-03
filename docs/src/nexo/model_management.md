# Model Management

This document describes how nexo-gateway and nexo-node cooperate to manage local LLM inference sessions: loading and unloading models, caching reusable prompt prefills, and handling requests that arrive when no inference node is available.

---

## 1. Architecture: Embedded Inference via nexo-ai

nexo-node runs inference **directly in-process** using the `nexo-ai` crate and the [Candle](https://github.com/huggingface/candle) ML framework. There are no external inference servers (llama-server, whisper-server, etc.) — all model execution happens within the nexo-node process.

```
nexo-node → nexo-ai::Coordinator → model traits → Candle inference (CPU/Metal GPU)
```

The `Coordinator` manages model slots: loading models into memory, unloading them, and routing inference requests to the appropriate model based on its category (chat, tool, image, listen, talk, etc.).

### Model categories

nexo-ai organizes models into capability categories:

| Category | Description | Primary model |
|----------|-------------|---------------|
| Chat | Text generation and conversation | Gemma 4 27B |
| Tool | Function calling with structured output | Gemma 4 27B |
| Image | Image analysis and understanding | Gemma 4 27B |
| Imagine | Image generation | Flux Schnell |
| Listen | Speech-to-text | Whisper Large v3 Turbo |
| Talk | Text-to-speech | Parler TTS |
| Embed | Text embeddings | Qwen3 Embed |

### Non-blocking inference

Candle inference is CPU/GPU-bound and synchronous. nexo-node bridges this with the async WebSocket transport using:

1. **Split WebSocket**: The connection is split into independent read/write halves.
2. **`tokio::select!`**: Multiplexes incoming WS frames and inference completion.
3. **`spawn_blocking`**: Inference runs on tokio's blocking thread pool, keeping the WS read loop responsive for ticks, pings, and new requests.

This ensures the node can process gateway heartbeats during multi-minute inference runs.

---

## 2. Model Lifecycle

### Declaration at connect time

When nexo-node starts, it scans `~/.nexo/local_models/` for downloaded model files and reports available models to the gateway:

```toml
# Auto-detected from disk — not manually configured
available_models = ["gemma4-27b", "flux-schnell", "whisper-large-v3-turbo"]
```

### Model load / unload

When an agent run requests a specific `model_id`, the gateway's loop runner calls `ensure_model_loaded`:

1. **Already in VRAM** — if `loaded_models[node]` already equals `model_id`, the run is routed immediately.
2. **On disk but not loaded** — the gateway sends `Method::ModelLoad` to the capable node and waits up to 300 seconds.
3. **Previous model loaded** — if the node has a different model in VRAM, the gateway first sends `Method::ModelUnload` (10 s timeout), then `Method::ModelLoad`.
4. **No eligible node** — the run is queued (see §4).

The node responds to each request and then pushes a `Method::ModelStatus` frame so the gateway's `loaded_models` map is always up to date.

### Protocol frames

| Direction | Method | Payload |
|-----------|--------|---------|
| Gateway → Node | `model.load` | `{ "modelId": "..." }` |
| Node → Gateway | Response | `{ "modelId": "...", "loaded": true, "error": null }` |
| Gateway → Node | `model.unload` | `{ "modelId": "..." }` |
| Node → Gateway | Response | `{ "unloaded": true }` |
| Node → Gateway | `model.status` (push) | `{ "loadedModelId": "...", "availableModels": [] }` |

---

## 3. Composable Prefills

A *prefill* is a system-level prompt prepended to every inference request. The prefill system is composable: individual **markdown files** are stored on disk and grouped into ordered **collections**. Collections are resolved at request time — their combined content is hashed, and only the hash is sent to the node, which caches content by hash.

### Storage layout

```
~/.nexo/storage/
  relational/
    gateway.db          # SQLite database
  markdown/
    <uuid>.md           # Raw markdown file content
```

SQLite holds metadata only: `id`, `category`, `description`, and `filename` per markdown file, plus an ordered list of file IDs per collection.

### Authoring markdown files

```json
{ "method": "prefill.markdown.create",
  "params": { "category": "identity", "description": "Core persona", "content": "# Identity\nYou are a helpful Rust assistant." } }

→ { "id": "01jx…" }
```

### Building a collection

```json
{ "method": "prefill.collection.create",
  "params": { "name": "default", "markdownIds": ["01jx…", "01jy…"] } }

→ { "id": "01jz…" }
```

### Attaching a collection to a session

```json
{ "method": "session.create",
  "params": { "name": "My assistant", "prefillCollectionId": "01jz…" } }

→ { "sessionId": "…", "prefillCollectionId": "01jz…" }
```

Sessions remember which collection to use; all agent runs on that session automatically receive the prefill.

### Prefill flow per agent run

```
Client                  Gateway                       Node
  │                        │                            │
  │── agent ──────────────►│                            │
  │  { prompt, sessionId } │                            │
  │                        │ resolve collection         │
  │                        │  read markdown files       │
  │                        │  join with "\n\n"          │
  │                        │  sha = SHA-256(combined)   │
  │                        │  cache sha → content       │
  │                        │                            │
  │                        │── agent ──────────────────►│
  │                        │  { messages, prefillSha }  │
  │                        │                            │ sha in cache?
  │                        │                            │  YES → use cached content
  │                        │                            │  NO  →
  │                        │◄── prefill.fetch ──────────│
  │                        │  { prefillSha }            │
  │                        │── response ───────────────►│
  │                        │  { content }               │
  │                        │                            │ cache sha → content
  │                        │                            │ prepend as system message
  │                        │                            │ run inference
  │                        │◄── agent response ─────────│
  │◄── agent event ────────│                            │
```

**SHA caching** means the node only fetches content once per unique collection state. On subsequent requests with the same collection and unchanged files, the node serves from its in-memory cache with no round-trip. The gateway also only reads markdown files once per agent run — the SHA is computed before the iteration loop and reused across all tool-call iterations.

**Cache invalidation**: the gateway's SHA→content cache is cleared whenever a markdown file or collection is deleted, ensuring nodes will re-fetch fresh content on the next request.

### Available methods

| Method | Description |
|--------|-------------|
| `prefill.markdown.create` | Store a new markdown file on disk |
| `prefill.markdown.list` | List all markdown files with metadata |
| `prefill.markdown.delete` | Delete a markdown file and its disk content |
| `prefill.collection.create` | Create an ordered collection of markdown file IDs |
| `prefill.collection.list` | List all collections with their ordered IDs |
| `prefill.collection.delete` | Delete a collection (items cascade) |
| `prefill.fetch` | (Node → Gateway) Fetch content by SHA-256 hash |

---

## 4. Request Queuing

When an agent run arrives but no suitable LLM node is available, the gateway queues the request instead of failing it.

### What happens

1. The run's `status` is set to `queued` in the `agent_runs` table.
2. The originating peer receives a `status: queued` event:

```json
{
  "event": "agent",
  "payload": {
    "runId": "...",
    "sessionId": "...",
    "status": "queued",
    "content": "No inference node is currently available. Your request has been queued and will be processed as soon as a node becomes available."
  }
}
```

### Drain on node connect

Whenever an LLM-capable node connects (or sends a `model.status` push that changes `loaded_models`), the gateway drains the queue:

1. Fetch all `queued` runs ordered by `queued_at ASC`.
2. Atomically claim each run (`UPDATE … WHERE status = 'queued'`).
3. Call `loop_runner::run` for each claimed run in order.

Double-processing is prevented because the `agent_task` is a single sequential async task — all `AgentCommand` variants are processed one at a time.

### Queued run columns

The `agent_runs` table stores the full request state needed for replay:

| Column | Purpose |
|--------|---------|
| `queued_at` | ISO-8601 timestamp used for ordering |
| `queued_prompt` | Original user prompt |
| `queued_context` | Optional JSON context blob |
| `queued_peer_id` | Peer that submitted the run |
| `model_id` | Requested model (may be `NULL`) |

> **Note**: The `thinking` flag is not preserved for queued runs — they replay with `thinking: false`.

---

## 5. Configuration

### nexo-ai.toml

```toml
# Model startup categories — loaded when nexo-node starts
startup_categories = ["chat", "talk"]

# Active model per category (used when no specific model_id is requested)
[active_models]
chat = "gemma4-27b"
imagine = "flux-schnell"

# Per-model overrides
[models.gemma4-27b]
temperature = 1.0
top_p = 0.95
top_k = 64
max_tokens = 4096

[models.flux-schnell]
default_steps = 4
default_guidance = 0.0
default_width = 1024
default_height = 1024
```

### nexo-node.toml

```toml
# URL of the nexo-gateway WebSocket
gateway_url = "ws://127.0.0.1:6969"

# Auto-populated based on downloaded models — no manual editing needed.
available_models = ["gemma4-27b"]
```

---

## 6. Model Downloads

nexo-node uses nexo-ai's download system to fetch model files (safetensors, tokenizer, config) from HuggingFace.

### Downloading a model

```bash
nexo-node models pull gemma4-27b            # download a specific model
nexo-node models pull all                   # download all registered models
nexo-node models list                       # show download status for all models
```

Models are stored at `~/.nexo/local_models/<model-name>/`. Model-specific files are stored directly; shared family files (e.g. tokenizers) are stored under `~/.nexo/local_models/shared/<family>/`.

### HuggingFace mirror

Downloads are routed through `https://hf-mirror.com` by default (firewall policy).
Set `HF_ENDPOINT` to override:

```bash
export HF_ENDPOINT=https://huggingface.co   # use primary HF server
nexo-node models pull gemma4-27b
```

For gated models, place a HuggingFace access token in `~/.nexo/hf_token.txt` or set `HF_TOKEN`.

### SHA-256 verification

On every `pull`, existing files are verified against the manifest's `sha256` field (when set).
Files that fail verification are automatically re-downloaded.

---

## 7. Gemma 4 Best Practices

Gemma 4 is the primary model family for chat, tool calling, and image analysis.

### Sampling configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | `1.0` | Do not lower — required for proper thinking mode |
| `top_p` | `0.95` | Nucleus sampling |
| `top_k` | `64` | Top-k filtering applied before top-p |

### Thinking mode

When `thinking: true` is set in the agent request:

1. `<|think|>` is prepended to the system prompt.
2. The model may emit `<|channel>thought\n...<channel|>` blocks in its response.
3. The gateway strips thinking blocks — only visible text is persisted.
4. Thinking content is sent ephemerally via the `thinkingContent` event field.

### Image analysis

For image analysis requests, Gemma 4 uses `temperature: 1.0` (previously 0.3). An optional `visualTokenBudget` parameter controls the number of visual tokens allocated for processing the image.
