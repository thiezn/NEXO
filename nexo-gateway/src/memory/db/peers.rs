use super::DbClient;
use crate::engine::PeerConnectionStateKind;
use crate::{Error, Result};
use nexo_core::{ModelId, Node, NodeStateKind, PeerId, User};

impl DbClient {
    /// Upsert a connected user row and mark it connected.
    ///
    /// # Arguments
    ///
    /// * `user` - The connected user whose identity, timestamps, and advertised tools should be persisted.
    pub async fn connect_user(&self, user: &User) -> Result {
        let client_id = user.id().client_id().to_string();
        let device_id = user.id().device_id().to_string();
        let connected_at = user.connected_at().to_rfc3339();
        let last_state_changed_at = Self::current_timestamp();

        sqlx::query(
            "INSERT INTO users (client_id, device_id, connection_state, first_connected_at, last_state_changed_at, connected_at)\n             VALUES (?, ?, ?, ?, ?, ?)\n             ON CONFLICT(client_id, device_id) DO UPDATE SET\n                 connection_state = excluded.connection_state,\n                 last_state_changed_at = excluded.last_state_changed_at,\n                 last_disconnected_at = NULL,\n                 connected_at = excluded.connected_at",
        )
        .bind(client_id)
        .bind(device_id)
        .bind(PeerConnectionStateKind::Connected.to_string())
        .bind(&connected_at)
        .bind(&last_state_changed_at)
        .bind(connected_at)
        .execute(self.pool())
        .await?;

        self.replace_user_tools(user.id(), &user.tools().iter().cloned().collect::<Vec<_>>())
            .await?;
        Ok(())
    }

    /// Mark a user row disconnected if it exists.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The user peer whose connection state should be marked as disconnected.
    pub async fn disconnect_user(&self, peer_id: PeerId) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "UPDATE users SET connection_state = ?, last_state_changed_at = ?, last_disconnected_at = ?\n             WHERE client_id = ? AND device_id = ?",
        )
        .bind(PeerConnectionStateKind::Disconnected.to_string())
        .bind(&now)
        .bind(&now)
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Load a persisted user by peer identifier.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The user peer to load from persistent storage.
    pub async fn get_user(&self, peer_id: PeerId) -> Result<User> {
        let row = sqlx::query_as::<_, User>(
            "SELECT\n                 users.client_id,\n                 users.device_id,\n                 users.connected_at,\n                 COALESCE(\n                     json_group_array(\n                         CASE WHEN tool_definitions.name IS NOT NULL THEN json_object(\n                             'name', tool_definitions.name,\n                             'description', tool_definitions.description,\n                             'parameters', json(tool_definitions.parameters_json),\n                             'contract_version', tool_definitions.contract_version,\n                             'execution', json(tool_definitions.execution_constraints_json)\n                         ) END
                    ) FILTER (WHERE tool_definitions.name IS NOT NULL),
                    '[]'
                 ) AS tools_json\n             FROM users\n             LEFT JOIN user_tools ON user_tools.user_client_id = users.client_id AND user_tools.user_device_id = users.device_id\n             LEFT JOIN tool_definitions ON tool_definitions.name = user_tools.tool_name\n             WHERE users.client_id = ? AND users.device_id = ?\n             GROUP BY users.client_id, users.device_id, users.connected_at",
        )
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "user",
            identifier: peer_id.to_string(),
        })
    }

    /// List all persisted users.
    pub async fn list_users(&self) -> Result<Vec<User>> {
        let rows = sqlx::query_as::<_, User>(
            "SELECT\n                 users.client_id,\n                 users.device_id,\n                 users.connected_at,\n                 COALESCE(\n                     json_group_array(\n                         CASE WHEN tool_definitions.name IS NOT NULL THEN json_object(\n                             'name', tool_definitions.name,\n                             'description', tool_definitions.description,\n                             'parameters', json(tool_definitions.parameters_json),\n                             'contract_version', tool_definitions.contract_version,\n                             'execution', json(tool_definitions.execution_constraints_json)\n                         ) END
                    ) FILTER (WHERE tool_definitions.name IS NOT NULL),
                    '[]'
                 ) AS tools_json\n             FROM users\n             LEFT JOIN user_tools ON user_tools.user_client_id = users.client_id AND user_tools.user_device_id = users.device_id\n             LEFT JOIN tool_definitions ON tool_definitions.name = user_tools.tool_name\n             GROUP BY users.client_id, users.device_id, users.connected_at\n             ORDER BY users.connected_at ASC",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows)
    }

    /// Upsert a connected node row and mark it connected.
    ///
    /// # Arguments
    ///
    /// * `node` - The connected node whose identity, state, tools, and model inventories should be persisted.
    pub async fn connect_node(&self, node: &Node) -> Result {
        let client_id = node.id().client_id().to_string();
        let device_id = node.id().device_id().to_string();
        let connected_at = node.connected_at().to_rfc3339();
        let node_state = NodeStateKind::from(node.state()).to_string();
        let last_state_changed_at = Self::current_timestamp();

        sqlx::query(
            "INSERT INTO nodes (client_id, device_id, connection_state, node_state, first_connected_at, last_state_changed_at, connected_at)\n             VALUES (?, ?, ?, ?, ?, ?, ?)\n             ON CONFLICT(client_id, device_id) DO UPDATE SET\n                 connection_state = excluded.connection_state,\n                 node_state = excluded.node_state,\n                 last_state_changed_at = excluded.last_state_changed_at,\n                 last_disconnected_at = NULL,\n                 connected_at = excluded.connected_at",
        )
        .bind(client_id)
        .bind(device_id)
        .bind(PeerConnectionStateKind::Connected.to_string())
        .bind(node_state)
        .bind(&connected_at)
        .bind(&last_state_changed_at)
        .bind(connected_at)
        .execute(self.pool())
        .await?;

        self.replace_node_tools(node.id(), &node.tools().iter().cloned().collect::<Vec<_>>())
            .await?;
        self.replace_node_models_on_disk(
            node.id(),
            node.models_on_disk()
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .await?;
        self.replace_node_models_in_memory(
            node.id(),
            node.models_in_memory()
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .await?;
        Ok(())
    }

    /// Mark a node row disconnected if it exists.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The node peer whose connection state should be marked as disconnected.
    pub async fn disconnect_node(&self, peer_id: PeerId) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "UPDATE nodes SET connection_state = ?, last_state_changed_at = ?, last_disconnected_at = ?\n             WHERE client_id = ? AND device_id = ?",
        )
        .bind(PeerConnectionStateKind::Disconnected.to_string())
        .bind(&now)
        .bind(&now)
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Load a persisted node by peer identifier.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The node peer to load from persistent storage.
    pub async fn get_node(&self, peer_id: PeerId) -> Result<Node> {
        let row = sqlx::query_as::<_, Node>(
            "SELECT\n                 nodes.client_id,\n                 nodes.device_id,\n                 nodes.node_state,\n                 nodes.connected_at,\n                 COALESCE(\n                     json_group_array(\n                         CASE WHEN tool_definitions.name IS NOT NULL THEN json_object(\n                             'name', tool_definitions.name,\n                             'description', tool_definitions.description,\n                             'parameters', json(tool_definitions.parameters_json),\n                             'contract_version', tool_definitions.contract_version,\n                             'execution', json(tool_definitions.execution_constraints_json)\n                         ) END
                    ) FILTER (WHERE tool_definitions.name IS NOT NULL),
                    '[]'
                 ) AS tools_json,\n                 COALESCE((SELECT json_group_array(model_id) FROM node_models_on_disk WHERE node_client_id = nodes.client_id AND node_device_id = nodes.device_id), '[]') AS models_on_disk_json,\n                 COALESCE((SELECT json_group_array(model_id) FROM node_models_in_memory WHERE node_client_id = nodes.client_id AND node_device_id = nodes.device_id), '[]') AS models_in_memory_json\n             FROM nodes\n             LEFT JOIN node_tools ON node_tools.node_client_id = nodes.client_id AND node_tools.node_device_id = nodes.device_id\n             LEFT JOIN tool_definitions ON tool_definitions.name = node_tools.tool_name\n             WHERE nodes.client_id = ? AND nodes.device_id = ?\n             GROUP BY nodes.client_id, nodes.device_id, nodes.node_state, nodes.connected_at",
        )
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "node",
            identifier: peer_id.to_string(),
        })
    }

    /// List all persisted nodes.
    pub async fn list_nodes(&self) -> Result<Vec<Node>> {
        let rows = sqlx::query_as::<_, Node>(
            "SELECT\n                 nodes.client_id,\n                 nodes.device_id,\n                 nodes.node_state,\n                 nodes.connected_at,\n                 COALESCE(\n                     json_group_array(\n                         CASE WHEN tool_definitions.name IS NOT NULL THEN json_object(\n                             'name', tool_definitions.name,\n                             'description', tool_definitions.description,\n                             'parameters', json(tool_definitions.parameters_json),\n                             'contract_version', tool_definitions.contract_version,\n                             'execution', json(tool_definitions.execution_constraints_json)\n                         ) END
                    ) FILTER (WHERE tool_definitions.name IS NOT NULL),
                    '[]'
                 ) AS tools_json,\n                 COALESCE((SELECT json_group_array(model_id) FROM node_models_on_disk WHERE node_client_id = nodes.client_id AND node_device_id = nodes.device_id), '[]') AS models_on_disk_json,\n                 COALESCE((SELECT json_group_array(model_id) FROM node_models_in_memory WHERE node_client_id = nodes.client_id AND node_device_id = nodes.device_id), '[]') AS models_in_memory_json\n             FROM nodes\n             LEFT JOIN node_tools ON node_tools.node_client_id = nodes.client_id AND node_tools.node_device_id = nodes.device_id\n             LEFT JOIN tool_definitions ON tool_definitions.name = node_tools.tool_name\n             GROUP BY nodes.client_id, nodes.device_id, nodes.node_state, nodes.connected_at\n             ORDER BY nodes.connected_at ASC",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows)
    }

    /// Update the current runtime state of a node.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The node peer whose runtime state should be updated.
    /// * `node_state` - The latest runtime state to persist for the node.
    pub async fn update_node_state(&self, peer_id: PeerId, node_state: NodeStateKind) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "UPDATE nodes SET node_state = ?, last_state_changed_at = ? WHERE client_id = ? AND device_id = ?",
        )
        .bind(node_state.to_string())
        .bind(now)
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Replace the set of models available on disk for the given node.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The node peer whose on-disk model inventory should be replaced.
    /// * `model_ids` - The complete set of models currently available on disk for the node.
    pub async fn replace_node_models_on_disk(
        &self,
        peer_id: PeerId,
        model_ids: &[ModelId],
    ) -> Result {
        let mut tx = self.pool().begin().await?;
        sqlx::query(
            "DELETE FROM node_models_on_disk WHERE node_client_id = ? AND node_device_id = ?",
        )
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        for model_id in model_ids {
            sqlx::query("INSERT INTO node_models_on_disk (node_client_id, node_device_id, model_id) VALUES (?, ?, ?)")
                .bind(peer_id.client_id().to_string())
                .bind(peer_id.device_id().to_string())
                .bind(String::from(*model_id))
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Replace the set of models currently loaded in memory for the given node.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The node peer whose in-memory model inventory should be replaced.
    /// * `model_ids` - The complete set of models currently loaded in memory for the node.
    pub async fn replace_node_models_in_memory(
        &self,
        peer_id: PeerId,
        model_ids: &[ModelId],
    ) -> Result {
        let mut tx = self.pool().begin().await?;
        sqlx::query(
            "DELETE FROM node_models_in_memory WHERE node_client_id = ? AND node_device_id = ?",
        )
        .bind(peer_id.client_id().to_string())
        .bind(peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        for model_id in model_ids {
            sqlx::query("INSERT INTO node_models_in_memory (node_client_id, node_device_id, model_id) VALUES (?, ?, ?)")
                .bind(peer_id.client_id().to_string())
                .bind(peer_id.device_id().to_string())
                .bind(String::from(*model_id))
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{
        ClientInfo, DeviceInfo, Node, NodeProperties, NodeState, User, UserProperties,
    };
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

    #[tokio::test]
    async fn connect_and_disconnect_user_updates_connection_state() {
        let db = test_db().await;
        let user = test_user();

        db.connect_user(&user).await.unwrap();
        db.disconnect_user(user.id()).await.unwrap();

        let (state,): (String,) = sqlx::query_as(
            "SELECT connection_state FROM users WHERE client_id = ? AND device_id = ?",
        )
        .bind(user.id().client_id().to_string())
        .bind(user.id().device_id().to_string())
        .fetch_one(db.pool())
        .await
        .unwrap();
        assert_eq!(state, PeerConnectionStateKind::Disconnected.to_string());
    }

    #[tokio::test]
    async fn connect_node_persists_runtime_state() {
        let db = test_db().await;
        let node = test_node();

        db.connect_node(&node).await.unwrap();
        db.update_node_state(node.id(), NodeStateKind::RunningInference)
            .await
            .unwrap();

        let (state,): (String,) =
            sqlx::query_as("SELECT node_state FROM nodes WHERE client_id = ? AND device_id = ?")
                .bind(node.id().client_id().to_string())
                .bind(node.id().device_id().to_string())
                .fetch_one(db.pool())
                .await
                .unwrap();
        assert_eq!(state, NodeStateKind::RunningInference.to_string());
    }

    #[tokio::test]
    async fn get_and_list_users_return_domain_users() {
        let db = test_db().await;
        let user = test_user();

        db.connect_user(&user).await.unwrap();

        let stored = db.get_user(user.id()).await.unwrap();
        let users = db.list_users().await.unwrap();

        assert_eq!(stored.id(), user.id());
        assert_eq!(stored.tools(), user.tools());
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].id(), user.id());
    }

    #[tokio::test]
    async fn get_and_list_nodes_return_domain_nodes() {
        let db = test_db().await;
        let node = test_node();

        db.connect_node(&node).await.unwrap();

        let stored = db.get_node(node.id()).await.unwrap();
        let nodes = db.list_nodes().await.unwrap();

        assert_eq!(stored.id(), node.id());
        assert_eq!(stored.state(), node.state());
        assert_eq!(stored.tools(), node.tools());
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id(), node.id());
    }
}
