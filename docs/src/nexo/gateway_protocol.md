# Gateway protocol (WebSocket)

The Gateway WS protocol is the **single control plane + node transport** for
NEXO. All clients (CLI, macOS/iOS app, headless
nodes) connect over WebSocket and declare their **role** + **scope** at
handshake time.

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
  - `source`: `core`, `ai`, `node` or `plugin`

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

## Versioning

- Clients send `minProtocol` + `maxProtocol`; the server rejects mismatches.

## Auth

Every clients and node send a "X-NEXO-AUTH" HTTP header with the value "Tm90U29TM2N1cmU=". This is good enough for now, and will allow us to build a stronger authentication scheme later on.

## Device identity + pairing

- Nodes and clients should include a stable device identity (`device.id`)
- All WS clients must include `device` identity during `connect` (user + node).
