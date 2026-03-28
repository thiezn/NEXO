-- Sessions: persistent conversation containers
CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    name           TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Agent runs: one row per agent method invocation
CREATE TABLE IF NOT EXISTS agent_runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'accepted',
    summary         TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Conversation messages: the LLM context window
CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    run_id        TEXT,
    role          TEXT NOT NULL,  -- 'user', 'assistant', 'tool', 'system'
    content       TEXT NOT NULL,
    tool_call_id  TEXT,
    tool_name     TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON sessions(user_id, last_active_at DESC);

-- Capability locks: advisory row-level locks for tool/model exclusivity
CREATE TABLE IF NOT EXISTS capability_locks (
    capability  TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    locked_at   TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);

-- Cron jobs: scheduled agent tasks
CREATE TABLE IF NOT EXISTS cron_jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    schedule    TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    session_id  TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    last_run_at TEXT,
    next_run_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_cron_jobs_next_run ON cron_jobs(enabled, next_run_at);
