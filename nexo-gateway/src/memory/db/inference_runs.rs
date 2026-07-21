use super::DbClient;
use super::db_types::InferenceRunRow;
use crate::agent::InferenceRunSnapshot;
use crate::{Error, Result};
use nexo_core::OperationId;

impl DbClient {
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

    /// Load the internal persisted row projection for an inference run.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation whose raw persisted row should be loaded.
    async fn get_inference_run_row(&self, operation_id: OperationId) -> Result<InferenceRunRow> {
        let row = sqlx::query_as::<_, InferenceRunRow>(
            "SELECT operation_id, run_state, node_client_id, node_device_id, model_id, unloading_model_id, error_message, created_at, preparing_started_at, node_selected_at, model_loading_started_at, in_progress_at, completed_at, failed_at, last_state_changed_at FROM inference_runs WHERE operation_id = ?",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        row.ok_or_else(|| Error::NotFound {
            resource: "inference_run",
            identifier: operation_id.to_string(),
        })
    }
}
