# Tools

NEXO is designed around tool use. Tools are callable functions that the agent can invoke during its reasoning loop. They come in two categories: **gateway-local tools** that run inside the gateway process, and **node tools** provided by connected nexo-node instances.

## Gateway-local tools

Gateway-local tools are always available — they don't depend on any node being connected. They execute directly in the gateway process and appear in the tool catalog with `source: "gateway"`.

All gateway-local tool output is cleaned up via a transformation pipeline: ANSI escape codes are stripped, long output is truncated, and content-specific transformations are applied (see [Output transformations](#output-transformations) below).

### IO tools

File, shell, and web access for the agent.

| Tool | Description |
|------|-------------|
| `io.read` | Read a file from the filesystem |
| `io.edit` | Edit an existing file or create a new one |
| `io.bash` | Execute a bash command on the gateway host |
| `io.web_fetch` | Fetch a web page or API endpoint |

#### `io.read`

Read file content with language-aware comment stripping.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | File path to read |
| `offset` | integer | no | Line number to start from (0-indexed) |
| `limit` | integer | no | Maximum number of lines to return |

The tool detects the programming language from the file extension and applies minimal filtering — stripping non-doc comments and normalizing blank lines. Data formats (JSON, YAML, TOML, etc.) pass through unchanged.

#### `io.edit`

Two modes: **create** (write a new file) or **edit** (find-and-replace in an existing file).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | File path to edit or create |
| `old_string` | string | no | Text to find and replace (edit mode) |
| `new_string` | string | no | Replacement text (edit mode) |
| `content` | string | no | Full file content (create mode) |

In edit mode, `old_string` must appear exactly once in the file. If it appears 0 or 2+ times, the tool returns an error asking for more context.

#### `io.bash`

Execute a shell command with timeout protection.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | string | yes | Bash command to execute |
| `timeout_ms` | integer | no | Timeout in milliseconds (default: 30000, max: 120000) |

Returns exit code, stdout, and stderr. Output is stripped of ANSI codes and truncated to 500 lines per stream.

#### `io.web_fetch`

Fetch a URL and return its content in a readable format.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | yes | URL to fetch |

Response handling by content type:
- **HTML** — converted to markdown
- **JSON** — compacted (values preserved, long strings truncated, arrays summarized)
- **Other** — returned as plain text with truncation

### Todo tools

TODO operations (`create`, `add`, `list`, `complete`, `delete`, `reorder`) are exposed as tools.

* These tools execute on the gateway (like other gateway-hosted tools).
* Results are appended to the transcript exactly like any other tool result.
* The gateway does **not** enforce TODO completion for run termination, it's up to the model to manage the TODO list and decide when it's "done."

### Notes tools

Git-backed note-taking for persistent memory across conversations.

| Tool | Description |
|------|-------------|
| `notes.create` | Create a timestamped markdown note |
| `notes.list` | List all saved note filenames |
| `notes.read` | Read a specific note by filename |
| `notes.update_summary` | Write the organized notes summary |

Notes are stored in `~/.nexo/nexo-storage/NOTES/` as git-tracked markdown files. The summary is at `NOTES/SUMMARY.md` and is updated periodically by a cron job.

## Node tools

Node tools are provided by external processes (nexo-node instances) that connect to the gateway over WebSocket. Each node registers its tools after connecting, and the gateway routes execution requests to the appropriate node.

### How it works

1. A **nexo-node** starts and connects to the gateway with `role: node`
2. After handshake, the node sends `tools.register` with full tool specifications (name, description, JSON Schema parameters)
3. The gateway stores these in an **in-memory registry** (not persisted to disk)
4. Clients and automated runs can query available tools via `tools.catalog`
5. Clients and automated runs can execute tools via `tools.execute` — the gateway routes the request to the owning node and relays the response

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
- `name`: Tool name (e.g. `io.read`, `notes.create`, `echo.run`)
- `description`: Human-readable description
- `source`: Where the tool comes from (`gateway` or `node`)
- `available`: Whether the tool is currently usable
- `parameters`: JSON Schema for the tool's input parameters

## Output transformations

Gateway-local tools apply a pipeline of transformations to clean up output for LLM consumption:

| Transformation | Applied to | Description |
|----------------|-----------|-------------|
| ANSI stripping | `io.read`, `io.bash` | Removes terminal color/style escape codes |
| Code filtering | `io.read` | Language-aware comment stripping, blank line normalization |
| Line truncation | `io.bash`, `io.web_fetch` | Keeps first/last portions of very long output |
| HTML→markdown | `io.web_fetch` | Converts HTML pages to readable markdown |
| JSON compaction | `io.web_fetch` | Summarizes JSON responses (truncated strings, array counts) |
