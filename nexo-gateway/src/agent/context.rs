use nexo_ws_schema::ToolEntry;
use sqlx::SqlitePool;

/// A message in the conversation context, ready for LLM consumption.
#[derive(Debug, Clone)]
pub struct ContextMessage {
    pub role: String,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

/// Load the full conversation history for a session, ordered by creation time.
pub async fn assemble(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Vec<ContextMessage>, sqlx::Error> {
    let rows: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT role, content, tool_call_id, tool_name
         FROM messages WHERE session_id = ? ORDER BY created_at ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(role, content, tool_call_id, tool_name)| ContextMessage {
            role,
            content,
            tool_call_id,
            tool_name,
        })
        .collect())
}

/// Build a system prompt section that describes available tools for the LLM.
pub fn build_tool_descriptions(tools: &[ToolEntry]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out = String::from("# Available Tools\n\n");
    for tool in tools.iter().filter(|t| t.available) {
        out.push_str(&format!("## {}\n", tool.name));
        out.push_str(&format!("{}\n", tool.description));
        if let Some(ref params) = tool.parameters {
            out.push_str(&format!(
                "Parameters: {}\n",
                serde_json::to_string(params).unwrap_or_default()
            ));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn assemble_empty_session(pool: SqlitePool) {
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

        let messages = assemble(&pool, "s1").await.unwrap();
        assert!(messages.is_empty());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn assemble_ordered_by_time(pool: SqlitePool) {
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

        // Insert messages with explicit timestamps to guarantee ordering
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, created_at)
             VALUES ('m1', 's1', 'user', 'first', '2026-01-01T00:00:01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, created_at)
             VALUES ('m2', 's1', 'assistant', 'second', '2026-01-01T00:00:02')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, created_at)
             VALUES ('m3', 's1', 'user', 'third', '2026-01-01T00:00:03')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let messages = assemble(&pool, "s1").await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "first");
        assert_eq!(messages[1].content, "second");
        assert_eq!(messages[2].content, "third");
    }

    #[test]
    fn build_tool_descriptions_empty() {
        let result = build_tool_descriptions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn build_tool_descriptions_formats_tools() {
        let tools = vec![
            ToolEntry {
                name: "echo.run".into(),
                description: "Echoes input".into(),
                source: "node".into(),
                available: true,
                parameters: Some(serde_json::json!({"type": "object"})),
            },
            ToolEntry {
                name: "offline.tool".into(),
                description: "Not available".into(),
                source: "node".into(),
                available: false,
                parameters: None,
            },
        ];
        let result = build_tool_descriptions(&tools);
        assert!(result.contains("echo.run"));
        assert!(result.contains("Echoes input"));
        // Offline tool should not be listed
        assert!(!result.contains("offline.tool"));
    }
}
