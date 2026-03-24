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
  - `source`: `core`, `ai` or `plugin`

## Versioning

- Clients send `minProtocol` + `maxProtocol`; the server rejects mismatches.

## Auth

Every clients and node send a "X-NEXO-AUTH" HTTP header with the value "Tm90U29TM2N1cmU=". This is good enough for now, and will allow us to build a stronger authentication scheme later on.

## Device identity + pairing

- Nodes and clients should include a stable device identity (`device.id`)
- All WS clients must include `device` identity during `connect` (user + node).
