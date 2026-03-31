-- Core identity

CREATE TABLE IF NOT EXISTS devices (
    id         TEXT PRIMARY KEY,
    role       TEXT NOT NULL,
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

-- Sessions and agent runs

CREATE TABLE IF NOT EXISTS sessions (
    id                    TEXT PRIMARY KEY,
    user_id               TEXT NOT NULL REFERENCES users(id),
    name                  TEXT,
    prefill_collection_id TEXT,
    model_id              TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'accepted',
    summary         TEXT,
    model_id        TEXT,
    queued_at       TEXT,
    queued_prompt   TEXT,
    queued_context  TEXT,
    queued_peer_id  TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    run_id       TEXT REFERENCES agent_runs(id),
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name    TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS capability_locks (
    capability TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES agent_runs(id),
    locked_at  TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    schedule    TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    session_id  TEXT REFERENCES sessions(id),
    enabled     INTEGER NOT NULL DEFAULT 1,
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

CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user     ON sessions(user_id, last_active_at DESC);
CREATE INDEX IF NOT EXISTS idx_cron_next_run     ON cron_jobs(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_queued ON agent_runs(status, queued_at) WHERE status = 'queued';
