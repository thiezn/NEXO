use super::DbClient;
use super::db_types::InferenceRunRecord;
use crate::{Error, Result};
use crate::agent::InferenceRunStateKind;
use nexo_core::{ModelId, OperationId, PeerId};

impl DbClient {
    /// Create or reset a run row at the preparing-context stage.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier whose run lifecycle should enter the preparing stage.
    pub async fn start_inference_run_preparing(&self, operation_id: OperationId) -> Result {
        let run_state = InferenceRunStateKind::PreparingContext.to_string();
        let now = Self::current_timestamp();
        sqlx::query(
            "INSERT INTO inference_runs (operation_id, run_state, created_at, preparing_started_at, last_state_changed_at)\n             VALUES (?, ?, ?, ?, ?)\n             ON CONFLICT(operation_id) DO UPDATE SET\n                run_state = excluded.run_state,\n                node_client_id = NULL,\n                node_device_id = NULL,\n                model_id = NULL,\n                error_message = NULL,\n                created_at = excluded.created_at,\n                preparing_started_at = excluded.preparing_started_at,\n                node_selected_at = NULL,\n                model_loading_started_at = NULL,\n                in_progress_at = NULL,\n                completed_at = NULL,\n                failed_at = NULL,\n                last_state_changed_at = excluded.last_state_changed_at",
        )
        .bind(operation_id.to_string())
        .bind(run_state)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Record the node and model chosen for the run and mark it unloading.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run being updated.
    /// * `node_peer_id` - The selected node for the run.
    /// * `model_id` - The model selected for the run.
    pub async fn mark_inference_run_unloading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> Result {
        self.update_run_state(
            operation_id,
            InferenceRunStateKind::UnloadingModel,
            Some(node_peer_id),
            Some(model_id),
            None,
            "node_selected_at",
        )
        .await
    }

    /// Record the node and model chosen for the run and mark it loading.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run being updated.
    /// * `node_peer_id` - The selected node for the run.
    /// * `model_id` - The model selected for the run.
    pub async fn mark_inference_run_loading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> Result {
        self.update_run_state(
            operation_id,
            InferenceRunStateKind::LoadingModel,
            Some(node_peer_id),
            Some(model_id),
            None,
            "model_loading_started_at",
        )
        .await
    }

    /// Record the node and model chosen for the run and mark it in progress.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run being updated.
    /// * `node_peer_id` - The selected node for the run.
    /// * `model_id` - The model selected for the run.
    pub async fn mark_inference_run_in_progress(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> Result {
        self.update_run_state(
            operation_id,
            InferenceRunStateKind::InProgress,
            Some(node_peer_id),
            Some(model_id),
            None,
            "in_progress_at",
        )
        .await
    }

    /// Mark a run completed.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run to finalize.
    pub async fn mark_inference_run_completed(&self, operation_id: OperationId) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "UPDATE inference_runs SET run_state = ?, completed_at = ?, last_state_changed_at = ? WHERE operation_id = ?",
        )
        .bind(InferenceRunStateKind::Completed.to_string())
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Mark a run failed with an error message.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run to finalize.
    /// * `error_message` - The failure reason to persist for the run.
    pub async fn mark_inference_run_failed(
        &self,
        operation_id: OperationId,
        error_message: &str,
    ) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "UPDATE inference_runs SET run_state = ?, error_message = ?, failed_at = ?, last_state_changed_at = ? WHERE operation_id = ?",
        )
        .bind(InferenceRunStateKind::Failed.to_string())
        .bind(error_message)
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Load the current persisted run state.
    ///
    /// This method stays crate-local because it returns an internal DB projection.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier whose persisted run state should be loaded.
    async fn get_inference_run(&self, operation_id: OperationId) -> Result<InferenceRunRecord> {
        let row = sqlx::query_as::<_, InferenceRunRecord>(
            "SELECT operation_id, run_state, node_client_id, node_device_id, model_id, error_message FROM inference_runs WHERE operation_id = ?",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "inference_run",
            identifier: operation_id.to_string(),
        })
    }

    /// Update a run to a new lifecycle state and stamp the matching stage timestamp.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier for the run being updated.
    /// * `run_state` - The lifecycle state to persist.
    /// * `node_peer_id` - The optional selected node for the run.
    /// * `model_id` - The optional selected model for the run.
    /// * `error_message` - The optional failure detail to persist.
    /// * `timestamp_column` - The lifecycle timestamp column that should be stamped for this transition.
    async fn update_run_state(
        &self,
        operation_id: OperationId,
        run_state: InferenceRunStateKind,
        node_peer_id: Option<PeerId>,
        model_id: Option<ModelId>,
        error_message: Option<&str>,
        timestamp_column: &str,
    ) -> Result {
        let node_client_id = node_peer_id.map(|value| value.client_id().to_string());
        let node_device_id = node_peer_id.map(|value| value.device_id().to_string());
        let model_id = model_id.map(String::from);
        let now = Self::current_timestamp();
        match timestamp_column {
            "node_selected_at" => {
                sqlx::query(
                    "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, node_selected_at = ?, last_state_changed_at = ? WHERE operation_id = ?",
                )
                .bind(run_state.to_string())
                .bind(node_client_id)
                .bind(node_device_id)
                .bind(model_id)
                .bind(error_message)
                .bind(&now)
                .bind(&now)
                .bind(operation_id.to_string())
                .execute(self.pool())
                .await?;
            }
            "model_loading_started_at" => {
                sqlx::query(
                    "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, model_loading_started_at = ?, last_state_changed_at = ? WHERE operation_id = ?",
                )
                .bind(run_state.to_string())
                .bind(node_client_id)
                .bind(node_device_id)
                .bind(model_id)
                .bind(error_message)
                .bind(&now)
                .bind(&now)
                .bind(operation_id.to_string())
                .execute(self.pool())
                .await?;
            }
            "in_progress_at" => {
                sqlx::query(
                    "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, in_progress_at = ?, last_state_changed_at = ? WHERE operation_id = ?",
                )
                .bind(run_state.to_string())
                .bind(node_client_id)
                .bind(node_device_id)
                .bind(model_id)
                .bind(error_message)
                .bind(&now)
                .bind(&now)
                .bind(operation_id.to_string())
                .execute(self.pool())
                .await?;
            }
            other => panic!("unsupported inference run timestamp column: {other}"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use nexo_core::{
        ClientInfo, ConversationMessage, DeviceInfo, InferenceIntent, InferenceOperation,
        ModelCapability, ModelSelection, Node, NodeProperties, NodeState, OperationId,
        ReasoningSettings, SessionId, ToolChoice, User, UserProperties,
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
        let properties = UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_node() -> Node {
        let properties = NodeProperties::new(ClientInfo::new("test-node"), DeviceInfo::default(), "token");
        Node::from_properties(&properties, NodeState::Idle, HashSet::new())
    }

    fn test_intent(operation_id: OperationId) -> InferenceIntent {
        InferenceIntent {
            operation_id,
            session_id: SessionId::new(),
            model_selection: ModelSelection::Capabilities(vec![ModelCapability::TextGeneration]),
            operation: InferenceOperation::MultiModal(MultiModalPayload::new_round(
                vec![ConversationMessage::new_text("hello")],
                Vec::new(),
                ToolChoice::Automatic,
                ReasoningSettings::default(),
            )),
        }
    }

    #[tokio::test]
    async fn run_state_transitions_persist_without_user_denormalization() {
        let db = test_db().await;
        let user = test_user();
        let node = test_node();
        let operation_id = OperationId::new();
        let intent = test_intent(operation_id);

        db.connect_user(&user).await.unwrap();
        db.connect_node(&node).await.unwrap();
        db.create_operation(operation_id, user.id()).await.unwrap();
        db.upsert_inference_intent(&intent).await.unwrap();
        db.start_inference_run_preparing(operation_id).await.unwrap();
        db.mark_inference_run_loading(operation_id, node.id(), ModelId::Kokoro82m)
            .await
            .unwrap();

        let run = db.get_inference_run(operation_id).await.unwrap();
        assert_eq!(run.run_state, InferenceRunStateKind::LoadingModel);
        assert_eq!(run.node_peer_id, Some(node.id()));
        assert_eq!(run.model_id, Some(ModelId::Kokoro82m));

        let columns: Vec<(String,)> = sqlx::query_as("SELECT name FROM pragma_table_info('inference_runs')")
            .fetch_all(db.pool())
            .await
            .unwrap();
        assert!(!columns.iter().any(|column| column.0 == "user_peer_id"));
    }
}
