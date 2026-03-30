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
         LEFT JOIN messages m ON m.session_id = s.id
         WHERE s.user_id = ?
         GROUP BY s.id
         ORDER BY s.last_active_at DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, name, created_at, last_active_at, count)| SessionEntry {
            session_id: id,
            name,
            created_at,
            last_active_at,
            message_count: count as u32,
        })
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

    let msg_rows: Vec<(String, String, String, String, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT id, role, content, created_at, tool_call_id, tool_name
             FROM messages WHERE session_id = ? ORDER BY created_at ASC",
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
    sqlx::query("DELETE FROM messages WHERE session_id = ?")
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
    let id = Frame::new_id();
    sqlx::query(
        "INSERT INTO messages (id, session_id, run_id, role, content, tool_call_id, tool_name)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(session_id)
    .bind(run_id)
    .bind(role)
    .bind(content)
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

/// Create a new agent run record.
pub async fn create_run(
    pool: &SqlitePool,
    run_id: &str,
    session_id: &str,
    idempotency_key: &str,
    model_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO agent_runs (id, session_id, idempotency_key, model_id) VALUES (?, ?, ?, ?)",
    )
    .bind(run_id)
    .bind(session_id)
    .bind(idempotency_key)
    .bind(model_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Mark an agent run as finished with a status and optional summary.
pub async fn finish_run(
    pool: &SqlitePool,
    run_id: &str,
    status: AgentStatus,
    summary: Option<&str>,
) -> Result<(), sqlx::Error> {
    let status_str =
        serde_json::to_value(status).ok().and_then(|v| v.as_str().map(String::from));
    sqlx::query(
        "UPDATE agent_runs SET status = ?, summary = ?, finished_at = datetime('now') WHERE id = ?",
    )
    .bind(status_str.as_deref().unwrap_or("failed"))
    .bind(summary)
    .bind(run_id)
    .execute(pool)
    .await?;
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

        let (name,): (Option<String>,) =
            sqlx::query_as("SELECT name FROM sessions WHERE id = ?")
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

        let (sid, _) = create_session(&pool, "u1", Some("chat"), None).await.unwrap();
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
        create_run(&pool, "run-1", &sid, "idem-1", None).await.unwrap();
        insert_message(&pool, &sid, Some("run-1"), "user", "hello", None, None)
            .await
            .unwrap();

        let cleared = clear_session(&pool, &sid).await.unwrap();
        assert!(cleared);

        // Verify everything is gone
        let (msg_count,): (i32,) =
            sqlx::query_as("SELECT COUNT(*) FROM messages WHERE session_id = ?")
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

        let (sess_count,): (i32,) =
            sqlx::query_as("SELECT COUNT(*) FROM sessions WHERE id = ?")
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
        create_run(&pool, "run-1", &sid, "idem-1", None).await.unwrap();

        let (status,): (String,) =
            sqlx::query_as("SELECT status FROM agent_runs WHERE id = 'run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "accepted");

        finish_run(&pool, "run-1", AgentStatus::Completed, Some("All done"))
            .await
            .unwrap();

        let (status, summary, finished): (String, Option<String>, Option<String>) =
            sqlx::query_as(
                "SELECT status, summary, finished_at FROM agent_runs WHERE id = 'run-1'",
            )
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "completed");
        assert_eq!(summary.as_deref(), Some("All done"));
        assert!(finished.is_some());
    }
}
