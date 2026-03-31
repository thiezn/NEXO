# Agent Loop

The agent loop is the brain of the NEXO system. It receives structured requests from
the gateway WebSocket, orchestrates inference and tool execution, and streams lifecycle
events back to the client.

## Overview

Each agent invocation spawns a **serialized loop** bound to a session. The loop:

1. Persists the user message
2. Assembles conversation context from the session's message history
3. Forwards an inference request to an LLM-capable node
4. Handles the response — either streaming a reply or executing tool calls
5. Repeats (tool results feed back into the next inference turn)
6. Marks the run as completed or failed

## Agent loop lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Brain
    participant Node

    Client->>Gateway: request:agent {prompt, sessionId?}
    Gateway->>Brain: AgentCommand::RunAgent
    Gateway-->>Client: response:agent {runId, sessionId, status:"accepted"}

    Brain->>Brain: persist user message
    Brain-->>Client: event:agent {status:"thinking"}
    Brain->>Brain: assemble context
    Brain->>Brain: acquire capability lock
    Brain->>Node: forward inference request
    Node-->>Brain: inference response (text or tool_calls)

    alt Plain reply
        Brain->>Brain: persist assistant message
        Brain-->>Client: event:agent {status:"streaming", content}
        Brain-->>Client: event:agent {status:"completed"}
    else Tool calls
        Brain-->>Client: event:agent {status:"tool_call", toolName}
        Brain->>Node: tools.execute {tool, args}
        Node-->>Brain: tool result
        Brain->>Brain: persist tool result
        Brain->>Brain: re-enter loop (back to inference)
    end

    Brain->>Brain: release locks, finish run
```

## Agent run state machine

```mermaid
stateDiagram-v2
    [*] --> Accepted
    Accepted --> Thinking
    Thinking --> Streaming: plain reply
    Thinking --> ToolCall: tool_calls in response
    ToolCall --> Thinking: tool results fed back
    Streaming --> Completed
    Thinking --> Failed: error
    ToolCall --> Failed: error
    [*] --> Cancelled: client cancels
```

### Status values

| Status | Description |
|--------|-------------|
| `accepted` | Run created, queued for processing |
| `thinking` | Assembling context and waiting for inference |
| `tool_call` | Model requested a tool execution |
| `streaming` | Model is generating a reply |
| `completed` | Run finished successfully |
| `failed` | Run encountered an error |
| `cancelled` | Run was cancelled by the client |

## Sessions

Sessions are persistent conversation containers. A client can:

- **Create** a session explicitly via `session.create`, or let the gateway auto-create
  one when sending an `agent` request without a `sessionId`.
- **List** active sessions via `session.list` (returns message counts and timestamps).
- **Retrieve** a session's full message history via `session.get`.
- **Clear** a session and all its data via `session.clear`.

```mermaid
stateDiagram-v2
    [*] --> Created: session.create
    Created --> Active: agent request
    Active --> Active: more messages
    Active --> Cleared: session.clear
    Cleared --> [*]
```

Sessions persist across client reconnections. A client stores the `sessionId` locally
and provides it in subsequent `agent` requests to maintain conversational context.

## Capability locking

When the agent loop invokes a node capability (e.g. LLM inference or a tool), it
acquires an advisory lock in SQLite:

- **Acquire**: `INSERT OR IGNORE` on the `capability_locks` table with a 5-minute expiry.
- **Release**: `DELETE` after the operation completes.
- **Expiry**: Locks older than their `expires_at` are automatically reaped.

This prevents two concurrent agent runs from using the same capability simultaneously,
ensuring consistent tool/model access across the distributed node network.

## Cron jobs

Cron jobs are scheduled agent tasks stored in the database. Each job specifies:

- A **schedule** (cron expression)
- A **prompt** (what the agent should do)
- An optional **session_id** (to continue a conversation)

The cron scheduler runs as a background task, polling every 60 seconds for due jobs.
When a job fires, it submits an `AgentCommand` to the brain and emits a `cron` event.

## End-to-end chat flow

The following diagram shows the full lifecycle of a chat message originating from the
Moretimer app, flowing through the gateway and node, and streaming back to the client.

```mermaid
sequenceDiagram
    participant App as Moretimer App
    participant GW as Gateway
    participant DB as SQLite
    participant Node as LLM Node

    App->>GW: session.create {name}
    GW->>DB: INSERT session
    GW-->>App: {sessionId}
    App->>App: store sessionId on thread

    App->>GW: agent {prompt, sessionId}
    GW->>DB: INSERT run + user message
    GW-->>App: response {runId, sessionId, status: accepted}
    GW-->>App: event:agent {status: thinking}

    GW->>GW: ensure_model_loaded
    alt Model not in VRAM
        GW->>Node: model.load {modelId}
        Node-->>GW: {loaded: true}
        Node->>GW: model.status (push)
    end

    GW->>GW: load SOUL.md + resolve prefill (if collection set)
    GW->>GW: build system prompt (SOUL + prefill + tools)
    GW->>Node: agent {messages, tools}

    Node-->>GW: inference response

    alt Plain reply
        GW->>DB: INSERT assistant message
        GW-->>App: event:agent {status: streaming, content}
        GW-->>App: event:agent {status: completed}
        App->>App: update message bubble
    else Tool calls
        GW-->>App: event:agent {status: tool_call, toolName}
        alt Gateway-native tool
            GW->>GW: execute locally (e.g. notes.create)
        else Node tool
            GW->>Node: tools.execute {tool, args}
            Node-->>GW: tool result
        end
        GW->>DB: INSERT tool message
        GW->>Node: agent {messages + tool results}
        Node-->>GW: final response
        GW->>DB: INSERT assistant message
        GW-->>App: event:agent {status: streaming, content}
        GW-->>App: event:agent {status: completed}
    end

    Note over App: Later, on re-open
    App->>GW: session.get {sessionId}
    GW->>DB: SELECT messages
    GW-->>App: {messages}
```

### Queued requests

When no LLM node is available at the time of an agent request, the gateway queues the
run instead of failing it:

```mermaid
sequenceDiagram
    participant App as Moretimer App
    participant GW as Gateway
    participant DB as SQLite
    participant Node as LLM Node

    App->>GW: agent {prompt, sessionId}
    GW->>DB: INSERT run (status: accepted)
    GW-->>App: response {runId, status: accepted}
    GW->>GW: no LLM node available
    GW->>DB: UPDATE run (status: queued)
    GW-->>App: event:agent {status: queued}

    Note over Node: Node connects later
    Node->>GW: connect {models: [...]}
    GW->>GW: drain_queue
    GW->>DB: claim queued runs
    GW->>Node: model.load + agent (inference)
    Node-->>GW: response
    GW-->>App: event:agent {status: streaming, content}
    GW-->>App: event:agent {status: completed}
```

## System prompt composition

The system prompt is assembled from three sources, in order:

1. **SOUL.md** — Always loaded from `~/.nexo/nexo-storage/SOUL.md` (if it exists). Defines the agent's personality and persistent instructions.
2. **Prefill collection** — If the session has a `prefill_collection_id`, the gateway resolves it from `PREFILL/collections.json`, reads each referenced markdown file, and concatenates them.
3. **Tool descriptions** — A structured description of all available tools (both node-registered and gateway-native).

If none of these produce content, a default "You are a helpful assistant." prompt is used.

## Gateway-native tools

Some tools execute directly on the gateway rather than being forwarded to a node. These are registered at startup and appear in the tool catalog alongside node tools (with `source: "gateway"`).

Current gateway-native tools:

| Tool | Description |
|------|-------------|
| `notes.create` | Create a timestamped note in git storage |
| `notes.list` | List all note filenames |
| `notes.read` | Read a specific note |
| `notes.update_summary` | Write the notes summary |

When the agent loop encounters a tool call, it checks gateway tools first. If a match is found, the tool executes locally without any node involvement. Otherwise, the call is forwarded to the appropriate node.

## Context assembly

Before each inference call, the brain loads the full conversation history from the
`messages` table for the current session, ordered chronologically. The system prompt
(see above) is prepended as a system message.

The context window includes messages with roles:
- `user` — user prompts
- `assistant` — model responses
- `tool` — tool execution results
- `system` — context injections
