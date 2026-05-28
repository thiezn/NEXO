//! Session lifecycle queries and deletion helpers.

use nexo_ws_schema::{Frame, SessionEntry, SessionGetResponse};
use sqlx::SqlitePool;

type SessionListRow = (String, Option<String>, Option<String>, String, String, i32);

/// Create a new session for a user. Returns `(session_id, prompt_collection_id)`.
pub async fn create_session(
    pool: &SqlitePool,
    user_id: &str,
    name: Option<&str>,
    prompt_collection_id: Option<&str>,
) -> Result<(String, Option<String>), sqlx::Error> {
    let id = Frame::new_id();

    sqlx::query(
        "INSERT INTO sessions (id, user_id, name, prompt_collection_id) VALUES (?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(user_id)
    .bind(name)
    .bind(prompt_collection_id)
    .execute(pool)
    .await?;

    Ok((id, prompt_collection_id.map(String::from)))
}

/// List all sessions for a user, ordered by most recently active.
pub async fn list_sessions(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<Vec<SessionEntry>, sqlx::Error> {
    let rows: Vec<SessionListRow> = sqlx::query_as(
        "SELECT s.id, s.name, s.prompt_collection_id, s.created_at, s.last_active_at, COUNT(m.id) as message_count
         FROM sessions s
         LEFT JOIN conversation_entries m ON m.session_id = s.id
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
            |(id, name, prompt_collection_id, created_at, last_active_at, count)| SessionEntry {
                session_id: id,
                name,
                prompt_collection_id,
                created_at,
                last_active_at,
                message_count: count as u32,
            },
        )
        .collect())
}

/// Get a session with all persisted messages.
pub async fn get_session(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Option<SessionGetResponse>, sqlx::Error> {
    let session: Option<(String, Option<String>, Option<String>, String)> = sqlx::query_as(
        "SELECT id, name, prompt_collection_id, created_at FROM sessions WHERE id = ?",
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await?;

    let Some((id, name, prompt_collection_id, created_at)) = session else {
        return Ok(None);
    };

    let messages = super::load_conversation_messages(pool, session_id).await?;

    Ok(Some(SessionGetResponse {
        session_id: id,
        name,
        prompt_collection_id,
        messages,
        created_at,
    }))
}

/// Clear a session and all related rows. Returns `true` when the session existed.
pub async fn clear_session(pool: &SqlitePool, session_id: &str) -> Result<bool, sqlx::Error> {
    let mut tx = pool.begin().await?;

    // Delete in dependency order so the session is removed atomically.
    sqlx::query(
        "DELETE FROM run_summaries WHERE run_id IN (SELECT id FROM runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "DELETE FROM tool_traces WHERE run_id IN (SELECT id FROM runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM conversation_entries WHERE session_id = ?")
        .bind(session_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        "DELETE FROM run_rounds WHERE run_id IN (SELECT id FROM runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "DELETE FROM capability_locks WHERE run_id IN (SELECT id FROM runs WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM runs WHERE session_id = ?")
        .bind(session_id)
        .execute(&mut *tx)
        .await?;
    let result = sqlx::query("DELETE FROM sessions WHERE id = ?")
        .bind(session_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(result.rows_affected() > 0)
}
