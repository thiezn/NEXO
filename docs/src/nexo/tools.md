# Tools

NEXO is designed around tool use. It provides a few out of the box default tools, and has an in memory registry of other tools it can leverage.

## Base Categories

These categories are the base tools provided directly by the gateway. They are tools that interact with the core of the NEXO system itself and provide fundamental capabilities that other tools and agents can build on top of.

### Core tools

- Read, Write, Delete files, Execute bash commands

### Memory tools

- Store, recall, forget

### Schedule tools

- cron_add, cron_list, etc

## Node tools

Node tools are provided by external processes (nexo-node instances) that connect to the gateway over WebSocket. Each node registers its tools after connecting, and the gateway routes execution requests to the appropriate node.

### How it works

1. A **nexo-node** starts and connects to the gateway with `role: node`
2. After handshake, the node sends `tools.register` with full tool specifications (name, description, JSON Schema parameters)
3. The gateway stores these in an **in-memory registry** (not persisted to disk)
4. Users/agents can query available tools via `tools.catalog`
5. Users/agents can execute tools via `tools.execute` — the gateway routes the request to the owning node and relays the response

### Lifecycle

- **Registration**: Nodes register tools immediately after connecting. They may call `tools.register` again to update their tool set.
- **Deregistration**: When a node disconnects, the gateway automatically removes all its tools from the registry.
- **Reconnection**: Nodes automatically reconnect on disconnect and re-register their tools.
- **Multiple nodes**: Multiple nodes can be connected simultaneously. Each node can provide different tools, and the gateway routes to the correct node.

### Built-in node tools

The `nexo-node` binary ships with two built-in tools for testing:

- `echo.run` — Echoes the input back as output
- `ping` — Returns "pong", useful for testing connectivity

## Tool registry

The tool registry can be queried by clients to see which tools are currently available. It shows the status of the tools, including whether the providing node is currently connected.

The `tools.catalog` method returns entries with:
- `name`: Tool name (e.g. `echo.run`)
- `description`: Human-readable description
- `source`: Where the tool comes from (`core`, `ai`, `node`)
- `available`: Whether the providing node is currently connected
- `parameters`: JSON Schema for the tool's input parameters
