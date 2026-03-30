# Model Management

This document describes how nexo-gateway and nexo-node cooperate to manage local LLM inference sessions: loading and unloading models, caching reusable prompt prefills, and handling requests that arrive when no inference node is available.

---

## 1. Model Lifecycle

### Declaration at connect time

When a nexo-node connects, it declares which models are available on-disk in `ConnectParams.models`:

```toml
# ~/.nexo/node.toml
available_models = ["qwen3-30b-a3b", "qwen3-8b"]
```

The gateway stores this list in `GatewayState.available_models` and uses it to route model-specific requests.

### Model load / unload

When an agent run requests a specific `model_id`, the gateway's loop runner calls `ensure_model_loaded`:

1. **Already in VRAM** — if `loaded_models[node]` already equals `model_id`, the run is routed immediately.
2. **On disk but not loaded** — the gateway sends `Method::ModelLoad` to the capable node and waits up to 300 seconds.
3. **Previous model loaded** — if the node has a different model in VRAM, the gateway first sends `Method::ModelUnload` (10 s timeout), then `Method::ModelLoad`.
4. **No eligible node** — the run is queued (see §3).

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

## 2. Composable Prefills

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

## 3. Request Queuing

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

---

## 4. Configuration

### node.toml

```toml
# URL of the nexo-gateway WebSocket
gateway_url = "ws://127.0.0.1:6969"

# Models available on disk, declared to the gateway at connect time.
# This is auto-populated based on downloaded models.
available_models = ["qwen3.5-35b-ab3b"]
```

Inference server URLs default to localhost ports 8001–8004 and are not persisted to `node.toml`.

---

## 5. Model Downloads

nexo-node includes a built-in download manager for fetching GGUF model files from HuggingFace.

### Downloading a model

```bash
nexo-node models pull qwen3.5-35b-ab3b   # download the primary inference model
nexo-node models pull all                 # download all registered models
nexo-node models list                     # show download status for all models
```

Models are stored at `~/.nexo/models/<model-name>/`. The environment variable
`NEXO_NODE_MODELS_DIR` overrides this base path.

### HuggingFace mirror

Downloads are routed through `https://hf-mirror.com` by default (firewall policy).
Set `HF_ENDPOINT` to override:

```bash
export HF_ENDPOINT=https://huggingface.co   # use primary HF server
nexo-node models pull qwen3.5-35b-ab3b
```

For gated models, place a HuggingFace access token in `~/.nexo/hf_token.txt` or set `HF_TOKEN`.

### SHA-256 verification

On every `pull`, existing files are verified against the manifest's `sha256` field (when set).
Files that fail verification are automatically re-downloaded.

---

## 6. Inference Service Management

nexo-node starts and monitors local inference servers automatically on `nexo-node start`.

### llama-server (primary)

nexo-node manages `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) for chat and tool-calling inference.

**Installation** — install the binary once manually:

```bash
# 1. Download macOS binaries from https://github.com/ggml-org/llama.cpp/releases
# 2. Extract and place the binary:
mkdir -p ~/.nexo/inference_services/llama_cpp
cp llama-server ~/.nexo/inference_services/llama_cpp/llama-server
chmod +x ~/.nexo/inference_services/llama_cpp/llama-server
```

**Startup** — when `nexo-node start` is run:

1. Checks for the binary at `~/.nexo/inference_services/llama_cpp/llama-server`. If absent, prints the install instructions above and continues connecting to the gateway without inference capability.
2. Checks for the model GGUF at `~/.nexo/models/qwen3.5-35b-ab3b/`. If absent, prints `nexo-node models pull qwen3.5-35b-ab3b` and continues.
3. If both are present, spawns `llama-server` on port 8001 and waits up to 60 seconds for it to report healthy.

**Lifecycle monitoring** — a background task polls `http://127.0.0.1:8001/health` every 5 seconds. After 3 consecutive failures it stops and restarts llama-server with exponential backoff (5 s → 10 s → 20 s → … → 60 s max). The gateway is not involved in this restart cycle; inference requests will fail with HTTP errors until the service recovers.

**Shutdown** — llama-server is terminated (SIGTERM) when `nexo-node` exits.

### Supported inference services

| Port | Service | Model |
|------|---------|-------|
| 8001 | `llama-server` | **Qwen3.5-35B-AB3B** Q4\_K\_M (chat + tool calling) |
| 8002 | `mlx-tts-server` | Qwen3-TTS |
| 8003 | `whisper-server` | Whisper (STT) |
| 8004 | `vllm-mlx` | Qwen3.5-9B (image analysis) |
| — | `qwen-image-mps` subprocess | Qwen-Image (image generation) |

Ports 8002–8004 and the image subprocess are not yet managed by nexo-node and must be started separately.
