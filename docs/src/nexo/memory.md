# Memory

## Backends

- **Markdown files** — Unstructured storage, LLM and user friendly
- **SQLite database** — Structured storage, queryable, relational
- **In-memory** — Volatile storage, fast access, temporary (GatewayState)

## SQLite schema

The gateway's SQLite database (`~/.nexo/nexo.db`) stores persistent state:

### Core tables

| Table | Purpose |
|-------|---------|
| `devices` | Known devices (nodes + clients), with first/last seen timestamps |
| `users` | Known users, linked to a device |
| `idempotency_keys` | Deduplication cache for side-effecting methods |

### Session & agent tables

| Table | Purpose |
|-------|---------|
| `sessions` | Conversation containers, linked to a user |
| `agent_runs` | One row per agent invocation, tracks status and summary |
| `messages` | Conversation messages (user, assistant, tool, system), ordered by time |
| `capability_locks` | Advisory locks for tool/model exclusivity during agent runs |
| `cron_jobs` | Scheduled agent tasks with cron expressions |

### Data flow

```
User prompt
  → messages table (role: "user")
  → context assembly (SELECT * FROM messages WHERE session_id ORDER BY created_at)
  → LLM inference (via node)
  → messages table (role: "assistant" or "tool")
  → agent_runs table (status: "completed")
```

## Categories

- **Conversation / chat transcripts** — stored in `messages` table per session
- **Daily / session summaries** — stored as `agent_runs.summary`
- **Core / long-term facts** — Markdown files for unstructured context
