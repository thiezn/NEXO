# Gateway architecture

## Overview

- A single long‑lived **Gateway** is the heart of the system.
- Control-plane clients (macOS/iOS app, CLI) connect to the
  Gateway over **WebSocket** on the configured bind host (default
  `127.0.0.1:6969`).
- **Nodes** (macOS/iOS/headless) also connect over **WebSocket**, but
  declare `role: node` with explicit capabilities/commands.

## Components and flows

### Gateway (daemon)

- Maintains connections to clients and nodes
- Exposes a typed WS API (requests, responses, server‑push events).
- Validates inbound frames against JSON Schema.
- Emits events like `agent`, `chat`, `presence`, `health`, `heartbeat`, `cron`.

### Clients (macOS / iOS app / CLI)

- One WS connection per client.
- Provide a user identity in `connect` and client identity; pairing is **user‑based** (`role: "user"`), client identity is informational (e.g. `cli`, `iOS`, `macOS`). The `role` field is the sole discriminator between users and nodes.
- Send requests (`health`, `status`, `send`, `agent`, `system-presence`).
- Subscribe to events (`tick`, `agent`, `presence`, `shutdown`).

### Nodes (macOS / iOS / headless)

- Connect to the **same WS server** with `role: node`.
- Provide a device identity in `connect`; pairing is **device‑based** (role `node`)
- After handshake, register full tool specs via `tools.register`.
- Listen for `tools.execute` requests forwarded by the gateway.
- Handle `agent` inference requests from the gateway loop runner.
- Respond to `model.load` / `model.unload` commands; push `model.status` updates.
- Expose commands like `epub_extractor.*`, `echo.*`, `ping`.
- Automatically reconnect and re-register on disconnect.
- On macOS, nexo-node also starts and monitors local inference servers (llama-server, etc.). See [Model Management](/nexo/model_management.md) §6.

## Connection lifecycle (single client)

```mermaid
sequenceDiagram
    participant Client
    participant Gateway

    Client->>Gateway: request:connect
    Gateway-->>Client: response (ok)
    Note right of Gateway: or response error + close
    Note left of Client: payload=hello-ok<br>snapshot: presence + health

    Gateway-->>Client: event:presence
    Gateway-->>Client: event:tick

    Client->>Gateway: request:agent
    Gateway-->>Client: response:agent<br>ack {runId, status:"accepted"}
    Gateway-->>Client: event:agent<br>(streaming)
    Gateway-->>Client: response:agent<br>final {runId, status, summary}
```

## Node registration + tool execution

```mermaid
sequenceDiagram
    participant Node
    participant Gateway
    participant User

    Node->>Gateway: request:connect (role=node, capabilities)
    Gateway-->>Node: response:hello-ok
    Node->>Gateway: request:tools.register (full specs)
    Gateway-->>Node: response {registered: N}
    Note right of Gateway: Tools stored in memory registry

    User->>Gateway: request:tools.catalog
    Gateway-->>User: response {tools: [...]}

    User->>Gateway: request:tools.execute {tool, args}
    Gateway->>Node: forwarded:tools.execute {tool, args}
    Node-->>Gateway: response {success, output}
    Gateway-->>User: relayed response {success, output}

    Note over Node,Gateway: On disconnect: gateway deregisters<br>all tools from this node
```

## Brain (agent loop)

The gateway spawns an **AgentHandle** at startup — a background task that processes
agent commands sequentially. When a client sends an `agent` request:

1. The handler creates (or resumes) a session and an agent run in SQLite
2. It submits an `AgentCommand::RunAgent` to the brain via an mpsc channel
3. The brain runs the loop: context assembly → inference → tool calls → repeat
4. Lifecycle events (`thinking`, `tool_call`, `streaming`, `completed`) are broadcast
   to all connected clients via the event channel

```mermaid
flowchart LR
    Client -->|request:agent| Handler
    Handler -->|AgentCommand| Brain
    Brain -->|inference request| Node
    Node -->|response| Brain
    Brain -->|event:agent| Client
    Brain -->|tools.execute| Node
```

### Capability locking

When the brain invokes a node capability (inference or tool execution), it acquires
an advisory lock in the `capability_locks` SQLite table. This prevents concurrent
agent runs from using the same capability. Locks expire after 5 minutes for crash
recovery.

### Cron scheduler

A background scheduler task runs alongside the brain, checking every 60 seconds for
due cron jobs. When a job fires, it emits a `cron` event and submits an agent command
to the brain.

## Sessions

A client can have multiple sessions and maintains a `sessionId` locally. Sessions are
created explicitly via `session.create` or implicitly when an `agent` request is sent
without a `sessionId`. After a client restart, it can query `session.list` to resume
previous conversations.

## Wire protocol (summary)

- Transport: WebSocket, text frames with JSON payloads.
- First frame **must** be `connect`.

- After handshake:
  - Requests: `{type:"request", id, method, params}` → `{type:"response", id, ok, payload|error}`
  - Events: `{type:"event", event, payload, seq?, stateVersion?}`
- Idempotency keys are required for side‑effecting methods (`send`, `agent`) to
  safely retry; the server keeps a short‑lived dedupe cache.
- Nodes must include `role: "node"` plus capabilities/commands in `connect`.

## Pairing

- All WS clients (users + nodes) include a **device identity** on `connect`.
- Users also provide a **user identity** (role `user`)
- The gateway stores the device identity for nodes, and the user identity + device identity for clients, in it's persistent memory store, including the time it was first seen and last seen.

Details: [Gateway protocol](/nexo/gateway_protocol.md)
