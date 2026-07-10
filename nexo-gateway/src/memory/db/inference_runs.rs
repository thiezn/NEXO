use super::DbClient;
use super::db_types::InferenceRunRow;
use crate::{Error, Result};
use crate::agent::{InferenceRun, InferenceRunSnapshot, InferenceRunState};
use nexo_core::OperationId;

impl DbClient {
    /// Persist the current state of an inference run.
    ///
    /// # Arguments
    ///
    /// * `run` - The typed inference run state to persist.
    pub async fn save_inference_run<S>(&self, run: &InferenceRun<S>) -> Result
    where
        for<'a> InferenceRunState: From<&'a InferenceRun<S>>,
    {
        let state = InferenceRunState::from(run);
        self.save_inference_run_state(run.operation_id(), &state).await
    }

    /// Load the current persisted inference run snapshot.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier whose persisted run state should be loaded.
    pub async fn load_inference_run_snapshot(
        &self,
        operation_id: OperationId,
    ) -> Result<InferenceRunSnapshot> {
        let row = self.get_inference_run_row(operation_id).await?;
        InferenceRunSnapshot::try_from(row)
    }

    /// Persist an application-facing inference run state for an operation.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation whose state should be persisted.
    /// * `state` - The application-facing state snapshot to persist.
    async fn save_inference_run_state(
        &self,
        operation_id: OperationId,
        state: &InferenceRunState,
    ) -> Result {
        let now = Self::current_timestamp();
        match state {
            InferenceRunState::Queued => {
                sqlx::query(
                    "INSERT INTO inference_runs (operation_id, run_state, created_at, last_state_changed_at)\n                     VALUES (?, ?, ?, ?)\n                     ON CONFLICT(operation_id) DO UPDATE SET\n                        run_state = excluded.run_state,\n                        node_client_id = NULL,\n                        node_device_id = NULL,\n                        model_id = NULL,\n                        error_message = NULL,\n                        preparing_started_at = NULL,\n                        node_selected_at = NULL,\n                        model_loading_started_at = NULL,\n                        in_progress_at = NULL,\n                        completed_at = NULL,\n                        failed_at = NULL,\n                        last_state_changed_at = excluded.last_state_changed_at",
                )
                .bind(operation_id.to_string())
                .bind(state.kind().to_string())
                .bind(&now)
                .bind(&now)
                .execute(self.pool())
                .await?;
            }
            InferenceRunState::PreparingContext => {
                sqlx::query(
                    "INSERT INTO inference_runs (operation_id, run_state, created_at, preparing_started_at, last_state_changed_at)\n                     VALUES (?, ?, ?, ?, ?)\n                     ON CONFLICT(operation_id) DO UPDATE SET\n                        run_state = excluded.run_state,\n                        node_client_id = NULL,\n                        node_device_id = NULL,\n                        model_id = NULL,\n                        error_message = NULL,\n                        preparing_started_at = excluded.preparing_started_at,\n                        node_selected_at = NULL,\n                        model_loading_started_at = NULL,\n                        in_progress_at = NULL,\n                        completed_at = NULL,\n                        failed_at = NULL,\n                        last_state_changed_at = excluded.last_state_changed_at",
                )
                .bind(operation_id.to_string())
                .bind(state.kind().to_string())
                .bind(&now)
                .bind(&now)
                .bind(&now)
                .execute(self.pool())
                .await?;
            }
            InferenceRunState::UnloadingModel {
                node_peer_id,
                model_id,
            } => {
                self.require_existing_inference_run(operation_id).await?;
                self.execute_state_update(
                    operation_id,
                    state,
                    Some(*node_peer_id),
                    Some(*model_id),
                    None,
                    Some(("node_selected_at", now.as_str())),
                    &now,
                )
                .await?;
            }
            InferenceRunState::LoadingModel {
                node_peer_id,
                model_id,
            } => {
                self.require_existing_inference_run(operation_id).await?;
                self.execute_state_update(
                    operation_id,
                    state,
                    Some(*node_peer_id),
                    Some(*model_id),
                    None,
                    Some(("model_loading_started_at", now.as_str())),
                    &now,
                )
                .await?;
            }
            InferenceRunState::InProgress {
                node_peer_id,
                model_id,
            } => {
                self.require_existing_inference_run(operation_id).await?;
                self.execute_state_update(
                    operation_id,
                    state,
                    Some(*node_peer_id),
                    Some(*model_id),
                    None,
                    Some(("in_progress_at", now.as_str())),
                    &now,
                )
                .await?;
            }
            InferenceRunState::Completed {
                node_peer_id,
                model_id,
            } => {
                self.require_existing_inference_run(operation_id).await?;
                self.execute_state_update(
                    operation_id,
                    state,
                    Some(*node_peer_id),
                    Some(*model_id),
                    None,
                    Some(("completed_at", now.as_str())),
                    &now,
                )
                .await?;
            }
            InferenceRunState::Failed {
                error_message,
                node_peer_id,
                model_id,
            } => {
                self.require_existing_inference_run(operation_id).await?;
                self.execute_state_update(
                    operation_id,
                    state,
                    *node_peer_id,
                    *model_id,
                    Some(error_message.as_str()),
                    Some(("failed_at", now.as_str())),
                    &now,
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Load the internal persisted row projection for an inference run.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation whose raw persisted row should be loaded.
    async fn get_inference_run_row(&self, operation_id: OperationId) -> Result<InferenceRunRow> {
        let row = sqlx::query_as::<_, InferenceRunRow>(
            "SELECT operation_id, run_state, node_client_id, node_device_id, model_id, error_message, created_at, preparing_started_at, node_selected_at, model_loading_started_at, in_progress_at, completed_at, failed_at, last_state_changed_at FROM inference_runs WHERE operation_id = ?",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "inference_run",
            identifier: operation_id.to_string(),
        })
    }

    /// Ensure that a persisted inference run row already exists.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation that must already have a persisted run row.
    async fn require_existing_inference_run(&self, operation_id: OperationId) -> Result {
        let exists = sqlx::query_scalar::<_, i64>(
            "SELECT 1 FROM inference_runs WHERE operation_id = ? LIMIT 1",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        if exists.is_some() {
            Ok(())
        } else {
            Err(Error::NotFound {
                resource: "inference_run",
                identifier: operation_id.to_string(),
            })
        }
    }

    /// Persist a non-initial state update with the fixed column set for that stage.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation whose run row should be updated.
    /// * `state` - The application-facing run state to persist.
    /// * `node_peer_id` - The selected node to persist, if any.
    /// * `model_id` - The selected model to persist, if any.
    /// * `error_message` - The failure message to persist, if any.
    /// * `stage_timestamp` - The timestamp column and value to stamp for this stage.
    /// * `now` - The current timestamp used for `last_state_changed_at`.
    async fn execute_state_update(
        &self,
        operation_id: OperationId,
        state: &InferenceRunState,
        node_peer_id: Option<nexo_core::PeerId>,
        model_id: Option<nexo_core::ModelId>,
        error_message: Option<&str>,
        stage_timestamp: Option<(&str, &str)>,
        now: &str,
    ) -> Result {
        let node_client_id = node_peer_id.map(|value| value.client_id().to_string());
        let node_device_id = node_peer_id.map(|value| value.device_id().to_string());
        let model_id = model_id.map(String::from);

        let query = match stage_timestamp.map(|(column, _)| column) {
            Some("node_selected_at") => {
                "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, node_selected_at = ?, completed_at = NULL, failed_at = NULL, last_state_changed_at = ? WHERE operation_id = ?"
            }
            Some("model_loading_started_at") => {
                "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, model_loading_started_at = ?, completed_at = NULL, failed_at = NULL, last_state_changed_at = ? WHERE operation_id = ?"
            }
            Some("in_progress_at") => {
                "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, in_progress_at = ?, completed_at = NULL, failed_at = NULL, last_state_changed_at = ? WHERE operation_id = ?"
            }
            Some("completed_at") => {
                "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, completed_at = ?, failed_at = NULL, last_state_changed_at = ? WHERE operation_id = ?"
            }
            Some("failed_at") => {
                "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, error_message = ?, failed_at = ?, completed_at = NULL, last_state_changed_at = ? WHERE operation_id = ?"
            }
            _ => unreachable!("stage timestamp is required for state updates"),
        };

        let (_, stage_timestamp_value) = stage_timestamp.expect("stage timestamp is required");
        sqlx::query(query)
            .bind(state.kind().to_string())
            .bind(node_client_id)
            .bind(node_device_id)
            .bind(model_id)
            .bind(error_message)
            .bind(stage_timestamp_value)
            .bind(now)
            .bind(operation_id.to_string())
            .execute(self.pool())
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use crate::agent::{InferenceRun, InferenceRunState};
    use nexo_core::{
        ClientInfo, ConversationMessage, DeviceInfo, InferenceIntent, InferenceOperation,
        ModelCapability, ModelId, ModelSelection, Node, NodeProperties, NodeState, OperationId,
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
        let queued = InferenceRun::new(operation_id, user.id());
        db.save_inference_run(&queued).await.unwrap();

        let preparing = queued.into_preparing_context();
        db.save_inference_run(&preparing).await.unwrap();
        let loading = preparing.into_loading_model(node.id(), ModelId::Kokoro82m);
        db.save_inference_run(&loading).await.unwrap();

        let run = db.load_inference_run_snapshot(operation_id).await.unwrap();
        assert_eq!(run.operation_id, operation_id);
        assert_eq!(run.state.kind().to_string(), "loading_model");
    assert!(run.timeline.preparing_started_at.is_some());
        assert!(run.timeline.model_loading_started_at.is_some());

        let InferenceRunState::LoadingModel {
            node_peer_id,
            model_id,
        } = &run.state
        else {
            panic!("expected loading_model snapshot")
        };

        assert_eq!(*node_peer_id, node.id());
        assert_eq!(*model_id, ModelId::Kokoro82m);

        let columns: Vec<(String,)> = sqlx::query_as("SELECT name FROM pragma_table_info('inference_runs')")
            .fetch_all(db.pool())
            .await
            .unwrap();
        assert!(!columns.iter().any(|column| column.0 == "user_peer_id"));
    }
}
