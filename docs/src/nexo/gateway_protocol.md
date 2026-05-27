# Gateway protocol (WebSocket)

The Gateway WS protocol is the **single control plane + node transport** for NEXO. All clients (CLI, macOS/iOS app, headless nodes) connect over WebSocket and declare their **role** + **scope** at handshake time.

## Transport

- WebSocket, text frames with JSON payloads.
- First frame **must** be a `connect` request.

## Handshake (connect)

Client → Gateway:

```json
{
  "type": "request",
  "id": "…",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "cli",
      "version": "1.2.3",
      "platform": "macos"
    },
    "role": "user",
    "scopes": ["user.read", "user.write"],
    "capabilities": [],
    "commands": [],
    "locale": "en-US",
    "userAgent": "NEXO-cli/1.2.3",
    "device": {
      "id": "device_fingerprint",
    }
  }
}
```

Gateway → Client:

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": { "type": "hello-ok", "protocol": 3, "policy": { "tickIntervalMs": 15000 } }
}
```

### Node example

```json
{
  "type": "request",
  "id": "…",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "rust-node",
      "version": "1.2.3",
      "platform": "macos"
    },
    "role": "node",
    "scopes": [],
    "capabilities": ["game_extractor", "epub_extractor"],
    "commands": ["game_extractor.extract", "game_extractor.analyze", "epub_extractor.extract"],
    "locale": "en-US",
    "userAgent": "NEXO-rust-node/1.2.3",
    "device": {
      "id": "device_fingerprint",
    }
  }
}
```

## Framing

- **Request**: `{type:"request", id, method, params}`
- **Response**: `{type:"response", id, ok, payload|error}`
- **Event**: `{type:"event", event, payload, seq?, stateVersion?}`

Side-effecting methods require **idempotency keys** (see schema).

There is no peer-to-peer transport between clients. Any client-to-client traffic is
routed through the gateway over the same WebSocket protocol.

## Roles + scopes

### Roles

- `user` = control plane client (CLI/UI).
- `node` = capability host (game_extractor/text_to_speech).

### Scopes (user)

Common scopes:

- `user.read`
- `user.write`
- `user.admin`

### Capabilities/commands (node)

Nodes declare capability claims at connect time:

- `capabilities`: high-level capability categories.
- `commands`: command allowlist for invoke.

The Gateway treats these as **claims** and enforces server-side allowlists.


### User helper methods

- users may call `tools.catalog` (`user.read`) to fetch the runtime tool catalog for an
  agent. The response includes grouped tools and provenance metadata:
  - `source`: `gateway` or `node`

### User routing identity

For `role: "user"` connections, the gateway currently uses `client.id` as the routing
identity for directed client messaging and session ownership. If multiple user peers are
connected with the same `client.id`, all matching peers receive directed `message` events.
`device.id` still identifies the concrete device connection.

## Client messaging (`send`)

Clients send directed messages to other connected clients via the gateway. The request is
acknowledged immediately, and the recipient receives a `message` event. Delivery is
`true` when at least one connected recipient peer matched the target identity.

Client → Gateway:

```json
{
  "type": "request",
  "id": "…",
  "method": "send",
  "params": {
    "target": "bob",
    "payload": { "text": "hello" },
    "idempotencyKey": "key-789"
  }
}
```

Gateway → Client (ack):

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": { "delivered": true }
}
```

Gateway → Target client:

```json
{
  "type": "event",
  "event": "message",
  "payload": {
    "messageId": "…",
    "from": "alice",
    "target": "bob",
    "payload": { "text": "hello" }
  }
}
```

Notes:

- `target` matches the recipient user routing identity (`client.id` for `role: "user"`)
- the gateway does not create a client-to-client return channel; replies are separate `send` requests
- a successful response with `delivered: false` means no connected recipient matched the target

## Node tool registration (`tools.register`)

After connecting, nodes send a `tools.register` request to provide the gateway with
full tool specifications (name, description, JSON Schema parameters). The gateway stores
these in an in-memory registry and uses them to serve `tools.catalog` responses and route
`tools.execute` requests.

Node → Gateway:

```json
{
  "type": "request",
  "id": "…",
  "method": "tools.register",
  "params": {
    "tools": [
      {
        "name": "echo.run",
        "description": "Echoes the input back as output",
        "parameters": {
          "type": "object",
          "properties": {
            "input": { "type": "string", "description": "The text to echo back" }
          },
          "required": ["input"]
        }
      }
    ]
  }
}
```

Gateway → Node:

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": { "registered": 1 }
}
```

Nodes may call `tools.register` multiple times to update their tool set. When a node
disconnects, the gateway automatically deregisters all of its tools.

## Tool execution (`tools.execute`)

Users (or the agent) request tool execution via `tools.execute`. The gateway looks up the
tool in its registry, forwards the request to the owning node, and relays the response
back to the caller. Execution has a 30-second timeout.

User → Gateway:

```json
{
  "type": "request",
  "id": "…",
  "method": "tools.execute",
  "params": {
    "tool": "echo.run",
    "args": { "input": "hello" },
    "idempotencyKey": "key-123"
  }
}
```

Gateway → User (relayed from node):

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": {
    "success": true,
    "output": "hello",
    "error": null
  }
}
```

Error cases:
- `tool_not_found`: the tool name is not in the registry
- `tool_unavailable`: the node hosting the tool is disconnected
- `timeout`: the node did not respond within 30 seconds

## Runs

The public run lifecycle is split across `run.start`, `run.instructions.append`,
and `run.stop`. The gateway creates (or resumes) a session, acknowledges
immediately with `status: "accepted"`, then streams `run` events as the background
run task processes the request.

Client → Gateway:

```json
{
  "type": "request",
  "id": "…",
  "method": "run.start",
  "params": {
    "input": "Summarize today's news",
    "idempotencyKey": "key-456",
    "sessionId": "optional-session-id",
    "instructions": { "files": ["notes.md"] },
    "thinking": true,
    "modelId": "gemma4-27b"
  }
}
```

Gateway → Client (immediate ack):

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": {
    "runId": "run-uuid-v7",
    "sessionId": "session-uuid-v7",
    "status": "accepted"
  }
}
```

Gateway → Client (streaming events):

```json
{ "type": "event", "event": "run", "payload": { "runId": "…", "sessionId": "…", "status": "thinking" } }
{ "type": "event", "event": "run", "payload": { "runId": "…", "sessionId": "…", "status": "tool_call", "toolName": "echo.run", "toolCallId": "call-1" } }
{ "type": "event", "event": "run", "payload": { "runId": "…", "sessionId": "…", "status": "streaming", "content": "Here is the summary...", "thinkingContent": "Let me analyze the request..." } }
{ "type": "event", "event": "run", "payload": { "runId": "…", "sessionId": "…", "status": "completed" } }
```

### `run.start` params

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | yes | The user message |
| `idempotencyKey` | string | yes | Deduplication key |
| `sessionId` | string | no | Resume existing session |
| `instructions` | object | no | Additional structured instructions persisted into the transcript |
| `thinking` | bool | no | Enable extended thinking mode (default: false) |
| `modelId` | string | no | Request a specific model |

### `run` event payload

| Field | Type | Description |
|-------|------|-------------|
| `runId` | string | The run identifier |
| `sessionId` | string | The session identifier |
| `status` | string | Current status (see below) |
| `content` | string? | Visible reply text (on `streaming`) |
| `thinkingContent` | string? | Ephemeral thinking text (on `streaming`, when thinking enabled) |
| `toolName` | string? | Tool being called (on `tool_call`) |
| `toolCallId` | string? | Tool call identifier (on tool events) |
| `error` | string? | Error message (on `failed`) |

### `run` status values

`accepted` → `thinking` → `streaming` → `completed`
`accepted` → `thinking` → `tool_call` → `thinking` → ... → `completed`
Any state → `failed` (on error) or `cancelled` (client-initiated).

### `run.instructions.append`

Append structured instructions to an active run. The gateway persists the payload as an
`instruction` transcript entry and the next round will observe it.

```json
{
  "type": "request",
  "id": "…",
  "method": "run.instructions.append",
  "params": {
    "runId": "run-uuid-v7",
    "instructions": { "notes": ["daily.md"] }
  }
}
```

```json
{
  "type": "response",
  "id": "…",
  "ok": true,
  "payload": {
    "queued": true,
    "messageId": "msg-uuid-v7"
  }
}
```

### `run.stop`

Stop an active run.

```json
{ "type": "request", "id": "…", "method": "run.stop", "params": { "runId": "run-uuid-v7" } }
```

```json
{ "type": "response", "id": "…", "ok": true, "payload": { "stopped": true } }
```

## Sessions

### `session.create`

Create a new transcript session.

```json
{ "type": "request", "id": "…", "method": "session.create", "params": { "name": "My chat", "promptCollectionId": "default" } }
```
```json
{ "type": "response", "id": "…", "ok": true, "payload": { "sessionId": "…", "promptCollectionId": "default" } }
```

### `session.list`

List all sessions for the current user.

```json
{ "type": "request", "id": "…", "method": "session.list", "params": {} }
```
```json
{
  "type": "response", "id": "…", "ok": true,
  "payload": {
    "sessions": [
      { "sessionId": "…", "name": "My chat", "promptCollectionId": "default", "createdAt": "…", "lastActiveAt": "…", "messageCount": 12 }
    ]
  }
}
```

### `session.get`

Retrieve a session with its full transcript history.

```json
{ "type": "request", "id": "…", "method": "session.get", "params": { "sessionId": "…" } }
```
```json
{
  "type": "response", "id": "…", "ok": true,
  "payload": {
    "sessionId": "…", "name": "My chat", "promptCollectionId": "default", "createdAt": "…",
    "messages": [
      { "id": "…", "role": "user", "content": "hello", "kind": "user_input", "createdAt": "…" },
      { "id": "…", "role": "assistant", "content": "hi back", "kind": "assistant_response", "createdAt": "…" }
    ]
  }
}
```

### `session.clear`

Delete a session and all associated data (messages, runs).

```json
{ "type": "request", "id": "…", "method": "session.clear", "params": { "sessionId": "…" } }
```
```json
{ "type": "response", "id": "…", "ok": true, "payload": { "cleared": true } }
```

## Cron jobs

### `cron.create`

Create a scheduled run task.

```json
{
  "type": "request", "id": "…", "method": "cron.create",
  "params": { "name": "Daily summary", "schedule": "0 9 * * *", "input": "Summarize yesterday" }
}
```
```json
{ "type": "response", "id": "…", "ok": true, "payload": { "jobId": "…" } }
```

### `cron.list`

List all cron jobs.

```json
{ "type": "request", "id": "…", "method": "cron.list", "params": {} }
```
```json
{
  "type": "response", "id": "…", "ok": true,
  "payload": { "jobs": [{ "jobId": "…", "name": "Daily summary", "schedule": "0 9 * * *", "enabled": true }] }
}
```

### `cron.delete`

Delete a cron job.

```json
{ "type": "request", "id": "…", "method": "cron.delete", "params": { "jobId": "…" } }
```
```json
{ "type": "response", "id": "…", "ok": true, "payload": { "deleted": true } }
```

## Versioning

- Clients send `minProtocol` + `maxProtocol`; the server rejects mismatches.

## Auth

Every clients and node send a "X-NEXO-AUTH" HTTP header with the value "Tm90U29TM2N1cmU=". This is good enough for now, and will allow us to build a stronger authentication scheme later on.

## Device identity + pairing

- Nodes and clients should include a stable device identity (`device.id`)
- All WS clients must include `device` identity during `connect` (user + node).
