# Memory

## Backends

- **Git-backed storage** — Persistent, versioned storage at `~/.nexo/nexo-storage` using a bare git repository with SSH remote sync
- **SQLite database** — Structured storage for sessions, runs, messages, cron jobs
- **In-memory** — Volatile storage, fast access, temporary (GatewayState)

## Git storage (`~/.nexo/nexo-storage`)

The git-backed storage repository holds user-facing persistent data. Every write triggers a pull → write → commit → push cycle, keeping all data versioned and portable.

### Repository structure

```
~/.nexo/nexo-storage/
├── SOUL.md                  # Agent personality (always prepended to system prompt)
├── PREFILL/
│   ├── collections.json     # Collection definitions
│   ├── identity.md          # Example prefill markdown
│   └── skills.md            # Example prefill markdown
└── NOTES/
    ├── SUMMARY.md           # Auto-generated summary of all notes
    └── 2026-01-15T12-00-00.md  # Timestamped notes
```

### SOUL.md

A markdown file whose content is always prepended to the agent's system prompt. Use it to define personality, communication style, or persistent instructions.

### Prefill system (`PREFILL/`)

Composable context templates that can be attached to sessions:

- **Markdown files** — Individual `.md` files stored in `PREFILL/`
- **Collections** — Named groups of markdown files defined in `PREFILL/collections.json`

When a session has a `prefill_collection_id`, the gateway resolves the collection at inference time: reads each referenced markdown file in order, concatenates them, and includes the result in the system prompt (after SOUL.md, before tool descriptions).

`collections.json` schema:

```json
{
  "collections": [
    {
      "id": "default",
      "name": "Default Identity",
      "description": "Core personality",
      "markdown_files": ["identity.md", "skills.md"]
    }
  ]
}
```

### Notes system (`NOTES/`)

Agent-accessible note storage with gateway-native tools:

| Tool | Description |
|------|-------------|
| `notes.create` | Create a timestamped note |
| `notes.list` | List all note filenames |
| `notes.read` | Read a specific note |
| `notes.update_summary` | Write `NOTES/SUMMARY.md` |

A default cron job (`notes-summary`, every 6 hours) prompts the agent to read all notes and generate an organized summary.

## SQLite schema

The gateway's SQLite database stores structured operational state:

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
- **Core / long-term facts** — SOUL.md and prefill markdown files in git storage
- **Notes** — Timestamped agent-created notes in git storage
