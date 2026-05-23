//! Conversation history loading for model input.

use sqlx::SqlitePool;

/// A transcript message ready to be sent to the model as conversation context.
#[derive(Debug, Clone)]
pub struct ConversationContextMessage {
    /// The message role expected by the model backend.
    pub role: String,
    /// The serialized message content.
    pub content: String,
    /// Optional tool call ID associated with this message.
    pub tool_call_id: Option<String>,
    /// Optional tool name associated with this message.
    pub tool_name: Option<String>,
}

/// Load the full persisted conversation context for a session.
pub async fn load_conversation_context(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Vec<ConversationContextMessage>, sqlx::Error> {
    let rows: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT role, content, tool_call_id, tool_name
            FROM transcript_entries WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(role, content, tool_call_id, tool_name)| ConversationContextMessage {
                role,
                content,
                tool_call_id,
                tool_name,
            },
        )
        .collect())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_context_empty_session(pool: SqlitePool) {
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

        let messages = load_conversation_context(&pool, "s1").await.unwrap();
        assert!(messages.is_empty());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_context_ordered_by_time(pool: SqlitePool) {
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
            "INSERT INTO transcript_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m1', 's1', 'user', 'first', 'message', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO transcript_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m2', 's1', 'assistant', 'second', 'message', '2026-01-01T00:00:02')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO transcript_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m3', 's1', 'user', 'third', 'message', '2026-01-01T00:00:03')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let messages = load_conversation_context(&pool, "s1").await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "first");
        assert_eq!(messages[1].content, "second");
        assert_eq!(messages[2].content, "third");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn load_conversation_context_preserves_insert_order_for_same_timestamp(pool: SqlitePool) {
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
            "INSERT INTO transcript_entries (id, session_id, role, content, entry_kind, created_at)
               VALUES ('m1', 's1', 'assistant', '<|tool_call>call:io.bash{}<tool_call|>', 'tool_call_intent', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO transcript_entries (id, session_id, role, content, entry_kind, tool_call_id, tool_name, created_at)
               VALUES ('m2', 's1', 'tool', 'stdout: games', 'tool_result', 'call-1', 'io.bash', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let messages = load_conversation_context(&pool, "s1").await.unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[1].role, "tool");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(messages[1].tool_name.as_deref(), Some("io.bash"));
    }
}
