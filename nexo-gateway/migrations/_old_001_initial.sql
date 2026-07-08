-- Core identity

CREATE TABLE IF NOT EXISTS devices (
    id         TEXT PRIMARY KEY,
    role       TEXT NOT NULL CHECK (role IN ('user', 'node')),
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    device_id  TEXT NOT NULL REFERENCES devices(id),
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key        TEXT PRIMARY KEY,
    method     TEXT NOT NULL,
    response   TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

-- Sessions and runs

CREATE TABLE IF NOT EXISTS sessions (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL REFERENCES users(id),
    name                 TEXT,
    prompt_collection_id TEXT,
    -- model_id             TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'accepted' CHECK (status IN ('accepted', 'queued', 'thinking', 'tool_call', 'streaming', 'completed', 'failed', 'cancelled')),
    model_id        TEXT,
    reasoning       TEXT NOT NULL DEFAULT '{"thinking":"disabled"}' CHECK (
        json_valid(reasoning)
        AND json_extract(reasoning, '$.thinking') IN ('disabled', 'enabled')
        AND (
            json_type(reasoning, '$.effort') IS NULL
            OR json_extract(reasoning, '$.effort') IN ('low', 'medium', 'high')
        )
    ),
    queued_at       TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS run_rounds (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(id),
    round_index      INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'started' CHECK (status IN ('started', 'completed', 'failed', 'queued', 'cancelled')),
    selected_peer_id TEXT,
    model_id         TEXT,
    rationale        TEXT,
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at      TEXT,
    UNIQUE (run_id, round_index)
);

CREATE TABLE IF NOT EXISTS conversation_entries (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    run_id       TEXT REFERENCES runs(id),
    round_id     TEXT REFERENCES run_rounds(id),
    role         TEXT NOT NULL CHECK (role IN ('system', 'developer', 'user', 'assistant', 'tool')),
    content      TEXT NOT NULL,
    entry_kind   TEXT NOT NULL CHECK (entry_kind IN ('user_input', 'instruction', 'assistant_response', 'tool_call_intent', 'tool_result')),
    tool_call_id TEXT,
    tool_name    TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tool_traces (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    round_id     TEXT NOT NULL REFERENCES run_rounds(id),
    tool_call_id TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    arguments    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'started' CHECK (status IN ('started', 'completed', 'failed')),
    output       TEXT,
    error        TEXT,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS run_summaries (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    round_id     TEXT REFERENCES run_rounds(id),
    kind         TEXT NOT NULL CHECK (kind IN ('final_response', 'failure', 'cancelled', 'terminal_state')),
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS capability_locks (
    capability TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(id),
    locked_at  TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    schedule    TEXT NOT NULL,
    input       TEXT NOT NULL,
    session_id  TEXT REFERENCES sessions(id),
    enabled     INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    last_run_at TEXT,
    next_run_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_models (
    node_id  TEXT NOT NULL,
    model_id TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (node_id, model_id)
);

-- Indexes

CREATE INDEX IF NOT EXISTS idx_conversation_entries_session ON conversation_entries(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_run_rounds_run ON run_rounds(run_id, round_index);
CREATE INDEX IF NOT EXISTS idx_tool_traces_run ON tool_traces(run_id, round_id);
CREATE INDEX IF NOT EXISTS idx_run_summaries_run ON run_summaries(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user     ON sessions(user_id, last_active_at DESC);
CREATE INDEX IF NOT EXISTS idx_cron_next_run     ON cron_jobs(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_runs_queued ON runs(status, queued_at) WHERE status = 'queued';
