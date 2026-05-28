//! Conversation persistence helpers for sessions and runs.

use sqlx::SqlitePool;

use crate::agent::session;
use nexo_ws_schema::Frame;

/// Persisted conversation kind for user-authored input.
pub const ENTRY_USER_INPUT: &str = "user_input";
/// Persisted conversation kind for system/developer instructions.
pub const ENTRY_INSTRUCTION: &str = "instruction";
/// Persisted conversation kind for assistant-authored final text.
pub const ENTRY_ASSISTANT_RESPONSE: &str = "assistant_response";
/// Persisted conversation kind for assistant-emitted tool calls.
pub const ENTRY_TOOL_CALL_INTENT: &str = "tool_call_intent";
/// Persisted conversation kind for tool execution results.
pub const ENTRY_TOOL_RESULT: &str = "tool_result";

/// Insert a conversation entry and update the session's last-active timestamp.
#[expect(clippy::too_many_arguments)]
pub async fn insert_conversation_entry(
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
        "INSERT INTO conversation_entries (
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

/// Insert a conversation message and bump the session activity timestamp.
#[cfg(test)]
pub async fn insert_message(
    pool: &SqlitePool,
    session_id: &str,
    run_id: Option<&str>,
    role: &str,
    content: &str,
    tool_call_id: Option<&str>,
    tool_name: Option<&str>,
) -> Result<String, sqlx::Error> {
    let entry_kind = match role {
        "user" => ENTRY_USER_INPUT,
        "assistant" => ENTRY_ASSISTANT_RESPONSE,
        "system" => ENTRY_INSTRUCTION,
        "tool" => ENTRY_TOOL_RESULT,
        _ => ENTRY_ASSISTANT_RESPONSE,
    };

    insert_conversation_entry(
        pool,
        session_id,
        run_id,
        None,
        role,
        content,
        entry_kind,
        tool_call_id,
        tool_name,
    )
    .await
}

/// Append structured context for an active run and return the persisted message ID.
pub async fn append_run_context(
    pool: &SqlitePool,
    run_id: &str,
    context: &serde_json::Value,
) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let content = serde_json::to_string(context).unwrap_or_default();
    let message_id = session::insert_conversation_entry(
        pool,
        &session_id,
        Some(run_id),
        None,
        "system",
        &content,
        ENTRY_INSTRUCTION,
        None,
        None,
    )
    .await?;
    Ok(Some(message_id))
}
