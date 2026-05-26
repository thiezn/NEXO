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
├── PROMPTS/
│   ├── collections.json     # Prompt collection definitions
│   ├── identity.md          # Example prompt document
│   └── skills.md            # Example prompt document
└── NOTES/
    ├── SUMMARY.md           # Auto-generated summary of all notes
    └── 2026-01-15T12-00-00.md  # Timestamped notes
```

### Prompt library (`PROMPTS/`)

Composable prompt documents that can be attached to sessions:

- **Prompt documents** — Individual `.md` files stored in `PROMPTS/`
- **Prompt collections** — Named groups of prompt documents defined in `PROMPTS/collections.json`

When a session has a `prompt_collection_id`, the gateway resolves the collection at inference time: reads each referenced prompt document in order, concatenates them, and includes the result in the system prompt before tool descriptions.

`collections.json` schema:

```json
{
  "collections": [
    {
      "id": "default",
      "name": "Default Identity",
      "description": "Core personality",
      "documents": ["identity.md", "skills.md"]
    }
  ]
}
```

### Notes system (`NOTES/`)

Run-accessible note storage with gateway-native tools:

| Tool | Description |
|------|-------------|
| `notes.create` | Create a timestamped note |
| `notes.list` | List all note filenames |
| `notes.read` | Read a specific note |
| `notes.update_summary` | Write `NOTES/SUMMARY.md` |

A default cron job (`notes-summary`, every 6 hours) prompts the run loop to read all notes and generate an organized summary.

## SQLite schema

The gateway's SQLite database stores structured operational state:

### Core tables

| Table | Purpose |
|-------|---------|
| `devices` | Known devices (nodes + clients), with first/last seen timestamps |
| `users` | Known users, linked to a device |
| `idempotency_keys` | Deduplication cache for side-effecting methods |

### Session & run tables

| Table | Purpose |
|-------|---------|
| `sessions` | Transcript containers, linked to a user |
| `runs` | One row per run invocation, tracks lifecycle state for the run |
| `run_rounds` | One row per inference round within a run |
| `transcript_entries` | Transcript entries (user, assistant, tool, system), ordered by time |
| `tool_traces` | Detailed records for tool execution attempts within a round |
| `run_summaries` | Terminal summaries stored separately from run lifecycle metadata |
| `capability_locks` | Advisory locks for tool/model exclusivity during runs |
| `cron_jobs` | Scheduled run tasks with cron expressions |

### Data flow

```
User input
  → transcript_entries table (role: "user")
  → context assembly (SELECT * FROM transcript_entries WHERE session_id ORDER BY created_at)
  → LLM inference (via node)
  → run_rounds table (round status + rationale + selected peer)
  → transcript_entries table (role: "assistant" or "tool")
  → tool_traces table (per tool invocation)
  → runs table (status: "completed")
  → run_summaries table (final summary text)
```

## Categories

- **Transcript / chat history** — stored in `transcript_entries` per session
- **Daily / session summaries** — stored in `run_summaries`
- **Core / long-term facts** — Prompt documents in git storage
- **Notes** — Timestamped run-created notes in git storage
