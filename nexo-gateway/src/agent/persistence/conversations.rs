//! Conversation persistence helpers for sessions and runs.

use nexo_core::{
    ContentPart, ConversationMessage, MessageRole, MetadataMap, TextPart, ToolCall, ToolCallId,
    ToolResult, ToolResultContent, ToolResultStatus,
};
use nexo_ws_schema::Frame;
use sqlx::SqlitePool;

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

type ConversationRow = (String, String, String, Option<String>, Option<String>);

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

/// Load the full persisted conversation history for a session.
pub async fn load_conversation_messages(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Vec<ConversationMessage>, sqlx::Error> {
    let rows: Vec<ConversationRow> = sqlx::query_as(
        "SELECT role, content, entry_kind, tool_call_id, tool_name
            FROM conversation_entries WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|(role, content, entry_kind, tool_call_id, tool_name)| {
            conversation_message_from_row(role, content, entry_kind, tool_call_id, tool_name)
        })
        .collect::<Result<Vec<_>, sqlx::Error>>()
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

/// Append structured instructions for an active run and return the persisted message ID.
pub async fn append_run_instructions(
    pool: &SqlitePool,
    run_id: &str,
    instructions: &serde_json::Value,
) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let content = serde_json::to_string(instructions).unwrap_or_default();
    let message_id = insert_conversation_entry(
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

fn decode_role(role: String) -> Result<MessageRole, sqlx::Error> {
    serde_json::from_value(serde_json::Value::String(role))
        .map_err(|error| sqlx::Error::Decode(Box::new(error)))
}

fn conversation_message_from_row(
    role: String,
    content: String,
    entry_kind: String,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
) -> Result<ConversationMessage, sqlx::Error> {
    let role = decode_role(role)?;
    let parts = match entry_kind.as_str() {
        ENTRY_TOOL_CALL_INTENT => tool_call_parts(&content),
        ENTRY_TOOL_RESULT => tool_result_parts(content, tool_call_id, tool_name),
        _ => text_parts(content),
    };

    Ok(ConversationMessage {
        role,
        parts,
        metadata: MetadataMap::new(),
    })
}

fn text_parts(content: String) -> Vec<ContentPart> {
    vec![ContentPart::Text(TextPart { text: content })]
}

fn tool_call_parts(content: &str) -> Vec<ContentPart> {
    match serde_json::from_str::<Vec<ToolCall>>(content) {
        Ok(calls) if !calls.is_empty() => calls.into_iter().map(ContentPart::ToolCall).collect(),
        _ => text_parts(content.to_string()),
    }
}

fn tool_result_parts(
    content: String,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
) -> Vec<ContentPart> {
    let Some(tool_call_id) = tool_call_id else {
        return text_parts(content);
    };
    let Some(tool_name) = tool_name else {
        return text_parts(content);
    };

    let status = if content.starts_with("Error:") {
        ToolResultStatus::Failure
    } else {
        ToolResultStatus::Success
    };

    vec![ContentPart::ToolResult(ToolResult {
        tool_call_id: ToolCallId::from(tool_call_id),
        tool_name,
        status,
        content: ToolResultContent::Text(content),
    })]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_messages_empty_session(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('s1', 'u1')")
            .execute(&pool)
            .await
            .unwrap();

        let messages = load_conversation_messages(&pool, "s1").await.unwrap();
        assert!(messages.is_empty());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_messages_ordered_by_time(pool: SqlitePool) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('s1', 'u1')")
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            "INSERT INTO conversation_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m1', 's1', 'user', 'first', 'user_input', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO conversation_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m2', 's1', 'assistant', 'second', 'assistant_response', '2026-01-01T00:00:02')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO conversation_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m3', 's1', 'user', 'third', 'user_input', '2026-01-01T00:00:03')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let messages = load_conversation_messages(&pool, "s1").await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(message_text(&messages[0]), "first");
        assert_eq!(message_text(&messages[1]), "second");
        assert_eq!(message_text(&messages[2]), "third");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_messages_preserves_insert_order_for_same_timestamp(
        pool: SqlitePool,
    ) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('s1', 'u1')")
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            "INSERT INTO conversation_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m1', 's1', 'assistant', '<|tool_call>call:io.bash{}<tool_call|>', 'tool_call_intent', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO conversation_entries (id, session_id, role, content, entry_kind, tool_call_id, tool_name, created_at)
               VALUES ('m2', 's1', 'tool', 'stdout: games', 'tool_result', 'call-1', 'io.bash', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let messages = load_conversation_messages(&pool, "s1").await.unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[1].role, MessageRole::Tool);
        let ContentPart::ToolResult(result) = &messages[1].parts[0] else {
            panic!("expected tool result part");
        };
        assert_eq!(result.tool_call_id.as_str(), "call-1");
        assert_eq!(result.tool_name, "io.bash");
    }

    fn message_text(message: &ConversationMessage) -> &str {
        let ContentPart::Text(TextPart { text }) = &message.parts[0] else {
            panic!("expected text part");
        };
        text
    }
}
