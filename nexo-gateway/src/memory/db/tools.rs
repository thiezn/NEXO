use super::DbClient;
use crate::{Error, Result};
use nexo_core::{PeerId, ToolDefinition};

impl DbClient {
    /// Upsert shared tool definitions and replace the tools linked to a user.
    ///
    /// # Arguments
    ///
    /// * `user_peer_id` - The user peer whose tool assignments should be replaced.
    /// * `tools` - The complete set of tool definitions currently exposed by the user.
    pub async fn replace_user_tools(
        &self,
        user_peer_id: PeerId,
        tools: &[ToolDefinition],
    ) -> Result {
        let mut tx = self.pool().begin().await?;
        self.upsert_tool_definitions_tx(&mut tx, tools).await?;

        sqlx::query("DELETE FROM user_tools WHERE user_client_id = ? AND user_device_id = ?")
            .bind(user_peer_id.client_id().to_string())
            .bind(user_peer_id.device_id().to_string())
            .execute(&mut *tx)
            .await?;

        for tool in tools {
            sqlx::query("INSERT INTO user_tools (user_client_id, user_device_id, tool_name) VALUES (?, ?, ?)")
                .bind(user_peer_id.client_id().to_string())
                .bind(user_peer_id.device_id().to_string())
                .bind(&tool.name)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Upsert shared tool definitions and replace the tools linked to a node.
    ///
    /// # Arguments
    ///
    /// * `node_peer_id` - The node peer whose tool assignments should be replaced.
    /// * `tools` - The complete set of tool definitions currently exposed by the node.
    pub async fn replace_node_tools(
        &self,
        node_peer_id: PeerId,
        tools: &[ToolDefinition],
    ) -> Result {
        let mut tx = self.pool().begin().await?;
        self.upsert_tool_definitions_tx(&mut tx, tools).await?;

        sqlx::query("DELETE FROM node_tools WHERE node_client_id = ? AND node_device_id = ?")
            .bind(node_peer_id.client_id().to_string())
            .bind(node_peer_id.device_id().to_string())
            .execute(&mut *tx)
            .await?;

        for tool in tools {
            sqlx::query("INSERT INTO node_tools (node_client_id, node_device_id, tool_name) VALUES (?, ?, ?)")
                .bind(node_peer_id.client_id().to_string())
                .bind(node_peer_id.device_id().to_string())
                .bind(&tool.name)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Load a shared tool definition by its unique tool name.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name to look up in `tool_definitions`.
    pub async fn get_tool_definition(&self, name: &str) -> Result<ToolDefinition> {
        let row = sqlx::query_as::<_, ToolDefinition>(
            "SELECT name, description, parameters_json, contract_version, execution_constraints_json\n             FROM tool_definitions WHERE name = ?",
        )
        .bind(name)
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "tool_definition",
            identifier: name.to_string(),
        })
    }

    /// Upsert shared tool-definition rows inside an existing transaction.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction used to keep definition and assignment updates atomic.
    /// * `tools` - The tool definitions that should exist in `tool_definitions` after the call.
    async fn upsert_tool_definitions_tx(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        tools: &[ToolDefinition],
    ) -> Result {
        let now = Self::current_timestamp();
        for tool in tools {
            sqlx::query(
                "INSERT INTO tool_definitions (name, description, parameters_json, contract_version, execution_constraints_json, created_at, updated_at)\n                 VALUES (?, ?, ?, ?, ?, ?, ?)\n                 ON CONFLICT(name) DO UPDATE SET\n                    description = excluded.description,\n                    parameters_json = excluded.parameters_json,\n                    contract_version = excluded.contract_version,\n                    execution_constraints_json = excluded.execution_constraints_json,\n                    updated_at = excluded.updated_at",
            )
            .bind(&tool.name)
            .bind(&tool.description)
            .bind(serde_json::to_string(&tool.parameters)?)
            .bind(tool.contract_version.clone())
            .bind(serde_json::to_string(&tool.execution)?)
            .bind(&now)
            .bind(&now)
            .execute(&mut **tx)
            .await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{
        ClientInfo, DeviceInfo, Node, NodeProperties, NodeState, ToolDefinition,
        ToolExecutionConstraints, User, UserProperties,
    };
    use serde_json::json;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::collections::HashSet;

    async fn test_db() -> DbClient {
        let pool = SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let db = DbClient::from_pool(pool);
        db.initialize_schema().await.unwrap();
        db
    }

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_node() -> Node {
        let properties =
            NodeProperties::new(ClientInfo::new("test-node"), DeviceInfo::default(), "token");
        Node::from_properties(&properties, NodeState::Idle, HashSet::new())
    }

    fn test_tool() -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_string(),
            description: "Echo input".to_string(),
            parameters: json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            contract_version: Some("1".to_string()),
            execution: ToolExecutionConstraints::default_read_only(),
        }
    }

    #[tokio::test]
    async fn replacing_peer_tools_upserts_shared_definitions() {
        let db = test_db().await;
        let user = test_user();
        let node = test_node();
        let tool = test_tool();

        db.connect_user(&user).await.unwrap();
        db.connect_node(&node).await.unwrap();
        db.replace_user_tools(user.id(), &[tool.clone()])
            .await
            .unwrap();
        db.replace_node_tools(node.id(), &[tool.clone()])
            .await
            .unwrap();

        let stored = db.get_tool_definition(&tool.name).await.unwrap();
        assert_eq!(stored, tool);

        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM tool_definitions")
            .fetch_one(db.pool())
            .await
            .unwrap();
        assert_eq!(count, 1);
    }
}
