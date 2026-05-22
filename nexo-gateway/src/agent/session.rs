use nexo_ws_schema::{AgentStatus, ConversationMessage, Frame, SessionEntry, SessionGetResponse};
use sqlx::SqlitePool;

/// Create a new session for a user. Returns (session_id, prefill_collection_id).
pub async fn create_session(
    pool: &SqlitePool,
    user_id: &str,
    name: Option<&str>,
    prefill_collection_id: Option<&str>,
) -> Result<(String, Option<String>), sqlx::Error> {
    let id = Frame::new_id();

    sqlx::query(
        "INSERT INTO sessions (id, user_id, name, prefill_collection_id) VALUES (?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(user_id)
    .bind(name)
    .bind(prefill_collection_id)
    .execute(pool)
    .await?;

    Ok((id, prefill_collection_id.map(String::from)))
}

/// List all sessions for a user, ordered by most recently active.
pub async fn list_sessions(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<Vec<SessionEntry>, sqlx::Error> {
    let rows: Vec<(String, Option<String>, String, String, i32)> = sqlx::query_as(
        "SELECT s.id, s.name, s.created_at, s.last_active_at, COUNT(m.id) as message_count
         FROM sessions s
         LEFT JOIN transcript_entries m ON m.session_id = s.id
         WHERE s.user_id = ?
         GROUP BY s.id
         ORDER BY s.last_active_at DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(id, name, created_at, last_active_at, count)| SessionEntry {
                session_id: id,
                name,
                created_at,
                last_active_at,
                message_count: count as u32,
            },
        )
        .collect())
}

/// Get a session with all its messages.
pub async fn get_session(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Option<SessionGetResponse>, sqlx::Error> {
    let session: Option<(String, Option<String>, String)> =
        sqlx::query_as("SELECT id, name, created_at FROM sessions WHERE id = ?")
            .bind(session_id)
            .fetch_optional(pool)
            .await?;

    let Some((id, name, created_at)) = session else {
        return Ok(None);
    };

    let msg_rows: Vec<(
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
    )> = sqlx::query_as(
        "SELECT id, role, content, created_at, tool_call_id, tool_name
            FROM transcript_entries WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await?;
    let messages = msg_rows
        .into_iter()
        .map(
            |(mid, role, content, msg_created, tool_call_id, tool_name)| ConversationMessage {
                id: mid,
                role,
                content,
                created_at: msg_created,
                tool_call_id,
                tool_name,
            },
        )
        .collect();

    Ok(Some(SessionGetResponse {
        session_id: id,
        name,
        messages,
        created_at,
    }))
}

/// Clear (delete) a session and all its related data. Returns true if the session existed.
pub async fn clear_session(pool: &SqlitePool, session_id: &str) -> Result<bool, sqlx::Error> {
    // Delete in dependency order
    sqlx::query(
        "DELETE FROM run_summaries WHERE run_id IN (SELECT id FROM agent_runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(pool)
    .await?;
    sqlx::query(
        "DELETE FROM tool_traces WHERE run_id IN (SELECT id FROM agent_runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(pool)
    .await?;
    sqlx::query("DELETE FROM transcript_entries WHERE session_id = ?")
        .bind(session_id)
        .execute(pool)
        .await?;
    sqlx::query(
        "DELETE FROM agent_rounds WHERE run_id IN (SELECT id FROM agent_runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(pool)
    .await?;
    sqlx::query("DELETE FROM capability_locks WHERE run_id IN (SELECT id FROM agent_runs WHERE session_id = ?)")
        .bind(session_id)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM agent_runs WHERE session_id = ?")
        .bind(session_id)
        .execute(pool)
        .await?;
    let result = sqlx::query("DELETE FROM sessions WHERE id = ?")
        .bind(session_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Insert a transcript entry and update the session's last active timestamp.
#[allow(clippy::too_many_arguments)]
pub async fn insert_transcript_entry(
    pool: &SqlitePool,
    session_id: &str,
    run_id: Option<&str>,
    round_id: Option<&str>,
    role: &str,
    content: &str,
    entry_kind: &str,
    tool_call_id: Option<&str>,
    tool_name: Option<&str>,
) -> Result<String, sqlx::Error> {
    let id = Frame::new_id();
    sqlx::query(
        "INSERT INTO transcript_entries (
            id, session_id, run_id, round_id, role, content, entry_kind, tool_call_id, tool_name
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(session_id)
    .bind(run_id)
    .bind(round_id)
    .bind(role)
    .bind(content)
    .bind(entry_kind)
    .bind(tool_call_id)
    .bind(tool_name)
    .execute(pool)
    .await?;

    sqlx::query("UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?")
        .bind(session_id)
        .execute(pool)
        .await?;

    Ok(id)
}

/// Insert a conversation message. Also updates the session's last_active_at.
pub async fn insert_message(
    pool: &SqlitePool,
    session_id: &str,
    run_id: Option<&str>,
    role: &str,
    content: &str,
    tool_call_id: Option<&str>,
    tool_name: Option<&str>,
) -> Result<String, sqlx::Error> {
    insert_transcript_entry(
        pool,
        session_id,
        run_id,
        None,
        role,
        content,
        "message",
        tool_call_id,
        tool_name,
    )
    .await
}

/// Create a new agent run record.
pub async fn create_run(
    pool: &SqlitePool,
    run_id: &str,
    session_id: &str,
    idempotency_key: &str,
    model_id: Option<&str>,
    thinking: bool,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO agent_runs (id, session_id, idempotency_key, model_id, thinking)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(run_id)
    .bind(session_id)
    .bind(idempotency_key)
    .bind(model_id)
    .bind(if thinking { 1 } else { 0 })
    .execute(pool)
    .await?;
    Ok(())
}

/// Return the next round index that should be used for a run.
pub async fn next_round_index(pool: &SqlitePool, run_id: &str) -> Result<usize, sqlx::Error> {
    let (next_index,): (i64,) = sqlx::query_as(
        "SELECT COALESCE(MAX(round_index), 0) + 1 FROM agent_rounds WHERE run_id = ?",
    )
    .bind(run_id)
    .fetch_one(pool)
    .await?;

    Ok(next_index.max(1) as usize)
}

/// Create a new round record for a run and return the round ID.
pub async fn create_round(
    pool: &SqlitePool,
    run_id: &str,
    round_index: usize,
    model_id: Option<&str>,
) -> Result<String, sqlx::Error> {
    let round_id = Frame::new_id();
    sqlx::query("INSERT INTO agent_rounds (id, run_id, round_index, model_id) VALUES (?, ?, ?, ?)")
        .bind(&round_id)
        .bind(run_id)
        .bind(round_index as i64)
        .bind(model_id)
        .execute(pool)
        .await?;
    Ok(round_id)
}

/// Mark a round as finished with status, rationale, and selected peer information.
pub async fn finish_round(
    pool: &SqlitePool,
    round_id: &str,
    status: &str,
    rationale: Option<&str>,
    selected_peer_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE agent_rounds
         SET status = ?, rationale = ?, selected_peer_id = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status)
    .bind(rationale)
    .bind(selected_peer_id)
    .bind(round_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Create a tool trace entry for a round and return its identifier.
pub async fn create_tool_trace(
    pool: &SqlitePool,
    run_id: &str,
    round_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Result<String, sqlx::Error> {
    let trace_id = Frame::new_id();
    let arguments_json = serde_json::to_string(arguments).unwrap_or_default();
    sqlx::query(
        "INSERT INTO tool_traces (id, run_id, round_id, tool_call_id, tool_name, arguments)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&trace_id)
    .bind(run_id)
    .bind(round_id)
    .bind(tool_call_id)
    .bind(tool_name)
    .bind(arguments_json)
    .execute(pool)
    .await?;
    Ok(trace_id)
}

/// Finish a tool trace with output or error information.
pub async fn finish_tool_trace(
    pool: &SqlitePool,
    trace_id: &str,
    success: bool,
    output: Option<&str>,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    let status = if success { "completed" } else { "failed" };
    sqlx::query(
        "UPDATE tool_traces
         SET status = ?, output = ?, error = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status)
    .bind(output)
    .bind(error)
    .bind(trace_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Store a summary record for a run.
pub async fn store_run_summary(
    pool: &SqlitePool,
    run_id: &str,
    round_id: Option<&str>,
    kind: &str,
    content: &str,
) -> Result<String, sqlx::Error> {
    let summary_id = Frame::new_id();
    sqlx::query(
        "INSERT INTO run_summaries (id, run_id, round_id, kind, content)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&summary_id)
    .bind(run_id)
    .bind(round_id)
    .bind(kind)
    .bind(content)
    .execute(pool)
    .await?;
    Ok(summary_id)
}

/// Stop an active run and return its session ID when the stop was applied.
pub async fn stop_run(pool: &SqlitePool, run_id: &str) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM agent_runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let status = serde_json::to_value(AgentStatus::Cancelled)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "cancelled".to_string());
    let result = sqlx::query(
        "UPDATE agent_runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(status)
    .bind(run_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        return Ok(None);
    }

    Ok(Some(session_id))
}

/// Append structured context for an active run and return the persisted message ID.
pub async fn append_run_context(
    pool: &SqlitePool,
    run_id: &str,
    context: &serde_json::Value,
) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM agent_runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let content = serde_json::to_string(context).unwrap_or_default();
    let message_id = insert_transcript_entry(
        pool,
        &session_id,
        Some(run_id),
        None,
        "system",
        &content,
        "context_append",
        None,
        None,
    )
    .await?;
    Ok(Some(message_id))
}

/// Return whether a run has been marked as cancelled.
pub async fn is_run_cancelled(pool: &SqlitePool, run_id: &str) -> Result<bool, sqlx::Error> {
    let row: Option<(String,)> = sqlx::query_as("SELECT status FROM agent_runs WHERE id = ?")
        .bind(run_id)
        .fetch_optional(pool)
        .await?;

    Ok(matches!(
        row.as_ref().map(|(status,)| status.as_str()),
        Some("cancelled")
    ))
}

/// Mark an agent run as finished with a status and optional summary.
pub async fn finish_run(
    pool: &SqlitePool,
    run_id: &str,
    status: AgentStatus,
    summary: Option<&str>,
) -> Result<(), sqlx::Error> {
    let status_str = serde_json::to_value(status)
        .ok()
        .and_then(|v| v.as_str().map(String::from));
    let result = sqlx::query(
        "UPDATE agent_runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(status_str.as_deref().unwrap_or("failed"))
    .bind(run_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 1
        && let Some(summary_text) = summary
    {
        let kind = match status {
            AgentStatus::Completed => "final_response",
            AgentStatus::Failed => "failure",
            AgentStatus::Cancelled => "cancelled",
            _ => "terminal_state",
        };
        let _ = store_run_summary(pool, run_id, None, kind, summary_text).await;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn create_session_generates_uuid(pool: SqlitePool) {
        // Insert a user first (FK constraint)
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('user-1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "user-1", Some("test session"), None::<&str>)
            .await
            .unwrap();
        assert!(!sid.is_empty());

        let (name,): (Option<String>,) = sqlx::query_as("SELECT name FROM sessions WHERE id = ?")
            .bind(&sid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(name.as_deref(), Some("test session"));
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn list_sessions_returns_by_user(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u2', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        create_session(&pool, "u1", Some("s1"), None).await.unwrap();
        create_session(&pool, "u1", Some("s2"), None).await.unwrap();
        create_session(&pool, "u2", Some("s3"), None).await.unwrap();

        let u1_sessions = list_sessions(&pool, "u1").await.unwrap();
        assert_eq!(u1_sessions.len(), 2);

        let u2_sessions = list_sessions(&pool, "u2").await.unwrap();
        assert_eq!(u2_sessions.len(), 1);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn list_sessions_includes_message_count(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        insert_message(&pool, &sid, None, "user", "hello", None, None)
            .await
            .unwrap();
        insert_message(&pool, &sid, None, "assistant", "hi", None, None)
            .await
            .unwrap();

        let sessions = list_sessions(&pool, "u1").await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].message_count, 2);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn get_session_returns_messages(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", Some("chat"), None)
            .await
            .unwrap();
        insert_message(&pool, &sid, None, "user", "hello", None, None)
            .await
            .unwrap();
        insert_message(&pool, &sid, None, "assistant", "hi back", None, None)
            .await
            .unwrap();

        let resp = get_session(&pool, &sid).await.unwrap().unwrap();
        assert_eq!(resp.session_id, sid);
        assert_eq!(resp.name.as_deref(), Some("chat"));
        assert_eq!(resp.messages.len(), 2);
        assert_eq!(resp.messages[0].role, "user");
        assert_eq!(resp.messages[1].role, "assistant");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn get_session_not_found(pool: SqlitePool) {
        let resp = get_session(&pool, "nonexistent").await.unwrap();
        assert!(resp.is_none());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn clear_session_deletes_everything(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", None, false)
            .await
            .unwrap();
        insert_message(&pool, &sid, Some("run-1"), "user", "hello", None, None)
            .await
            .unwrap();

        let cleared = clear_session(&pool, &sid).await.unwrap();
        assert!(cleared);

        // Verify everything is gone
        let (msg_count,): (i32,) =
            sqlx::query_as("SELECT COUNT(*) FROM transcript_entries WHERE session_id = ?")
                .bind(&sid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(msg_count, 0);

        let (run_count,): (i32,) =
            sqlx::query_as("SELECT COUNT(*) FROM agent_runs WHERE session_id = ?")
                .bind(&sid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(run_count, 0);

        let (sess_count,): (i32,) = sqlx::query_as("SELECT COUNT(*) FROM sessions WHERE id = ?")
            .bind(&sid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(sess_count, 0);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn clear_nonexistent_returns_false(pool: SqlitePool) {
        let cleared = clear_session(&pool, "nonexistent").await.unwrap();
        assert!(!cleared);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn insert_message_updates_last_active(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();

        let (before,): (String,) =
            sqlx::query_as("SELECT last_active_at FROM sessions WHERE id = ?")
                .bind(&sid)
                .fetch_one(&pool)
                .await
                .unwrap();

        // Small delay so datetime('now') is different
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        insert_message(&pool, &sid, None, "user", "hello", None, None)
            .await
            .unwrap();

        let (after,): (String,) =
            sqlx::query_as("SELECT last_active_at FROM sessions WHERE id = ?")
                .bind(&sid)
                .fetch_one(&pool)
                .await
                .unwrap();

        assert!(after >= before);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn create_and_finish_run(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", None, false)
            .await
            .unwrap();

        let (status, thinking): (String, i64) =
            sqlx::query_as("SELECT status, thinking FROM agent_runs WHERE id = 'run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "accepted");
        assert_eq!(thinking, 0);

        finish_run(&pool, "run-1", AgentStatus::Completed, Some("All done"))
            .await
            .unwrap();

        let (status, finished): (String, Option<String>) =
            sqlx::query_as("SELECT status, finished_at FROM agent_runs WHERE id = 'run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "completed");
        assert!(finished.is_some());

        let (summary_kind, summary_content): (String, String) =
            sqlx::query_as("SELECT kind, content FROM run_summaries WHERE run_id = 'run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(summary_kind, "final_response");
        assert_eq!(summary_content, "All done");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn stop_run_marks_run_cancelled(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", None, false)
            .await
            .unwrap();

        let stopped_session = stop_run(&pool, "run-1").await.unwrap();
        assert_eq!(stopped_session.as_deref(), Some(sid.as_str()));
        assert!(is_run_cancelled(&pool, "run-1").await.unwrap());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn append_run_context_persists_system_message(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", None, false)
            .await
            .unwrap();

        let message_id =
            append_run_context(&pool, "run-1", &serde_json::json!({"hint": "use notes"}))
                .await
                .unwrap();
        assert!(message_id.is_some());

        let row: (String, String,) = sqlx::query_as(
            "SELECT role, content FROM transcript_entries WHERE run_id = 'run-1' ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.0, "system");
        assert!(row.1.contains("use notes"));
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn round_trace_and_summary_records_persist(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", Some("gemma"), true)
            .await
            .unwrap();
        let round_id = create_round(&pool, "run-1", 1, Some("gemma"))
            .await
            .unwrap();
        let trace_id = create_tool_trace(
            &pool,
            "run-1",
            &round_id,
            "call-1",
            "echo.run",
            &serde_json::json!({"input": "hello"}),
        )
        .await
        .unwrap();
        finish_tool_trace(&pool, &trace_id, true, Some("hello"), None)
            .await
            .unwrap();
        finish_round(
            &pool,
            &round_id,
            "completed",
            Some("Reasoning"),
            Some("node-1"),
        )
        .await
        .unwrap();
        store_run_summary(&pool, "run-1", Some(&round_id), "final_response", "hello")
            .await
            .unwrap();

        let (round_status, rationale, selected_peer_id): (String, Option<String>, Option<String>) =
            sqlx::query_as(
                "SELECT status, rationale, selected_peer_id FROM agent_rounds WHERE id = ?",
            )
            .bind(&round_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(round_status, "completed");
        assert_eq!(rationale.as_deref(), Some("Reasoning"));
        assert_eq!(selected_peer_id.as_deref(), Some("node-1"));

        let (tool_status, output): (String, Option<String>) =
            sqlx::query_as("SELECT status, output FROM tool_traces WHERE id = ?")
                .bind(&trace_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(tool_status, "completed");
        assert_eq!(output.as_deref(), Some("hello"));
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn next_round_index_starts_after_existing_rounds(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();

        let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
        create_run(&pool, "run-1", &sid, "idem-1", Some("gemma"), false)
            .await
            .unwrap();

        assert_eq!(next_round_index(&pool, "run-1").await.unwrap(), 1);

        create_round(&pool, "run-1", 1, Some("gemma"))
            .await
            .unwrap();

        assert_eq!(next_round_index(&pool, "run-1").await.unwrap(), 2);
    }
}
