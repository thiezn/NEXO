use super::DbClient;
use crate::Result;
use crate::agent::{
    AgentJobKind, AgentJobSchedulerState, InferenceRunStateKind, RunnableJobCandidate,
};
use nexo_core::{InferenceIntent, InferenceOperationKind, ModelId, OperationId, PeerId};

const RUNNABLE_JOB_BATCH_SIZE: i64 = 32;

impl DbClient {
    /// Enqueue a new inference job at the back of the FIFO queue.
    ///
    /// # Arguments
    ///
    /// * `user_peer_id` - The user peer that owns the operation.
    /// * `intent` - The canonical inference intent to persist for the job.
    pub async fn enqueue_inference_job(
        &self,
        user_peer_id: PeerId,
        intent: &InferenceIntent,
    ) -> Result<i64> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;

        sqlx::query(
            "INSERT INTO operations (operation_id, user_client_id, user_device_id, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(intent.operation_id.to_string())
        .bind(user_peer_id.client_id().to_string())
        .bind(user_peer_id.device_id().to_string())
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO inference_intents (operation_id, session_id, operation_kind, model_selection_json, intent_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(intent.operation_id.to_string())
        .bind(intent.session_id.to_string())
        .bind(InferenceOperationKind::from(&intent.operation).to_string())
        .bind(serde_json::to_string(&intent.model_selection)?)
        .bind(serde_json::to_string(intent)?)
        .bind(&now)
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO inference_runs (operation_id, run_state, created_at, last_state_changed_at) VALUES (?, ?, ?, ?)",
        )
        .bind(intent.operation_id.to_string())
        .bind(InferenceRunStateKind::Queued.to_string())
        .bind(&now)
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        let result = sqlx::query(
            "INSERT INTO agent_jobs (operation_id, job_kind, scheduler_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(intent.operation_id.to_string())
        .bind(AgentJobKind::RunInference.to_string())
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .bind(&now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(result.last_insert_rowid())
    }

    /// Load a bounded batch of due runnable jobs in stable FIFO order.
    pub(crate) async fn list_runnable_jobs(&self) -> Result<Vec<RunnableJobCandidate>> {
        sqlx::query_as::<_, RunnableJobCandidate>(
            "SELECT jobs.queue_position, jobs.operation_id, operations.user_client_id, operations.user_device_id, jobs.job_kind\n             FROM agent_jobs AS jobs\n             INNER JOIN operations ON operations.operation_id = jobs.operation_id\n             WHERE jobs.scheduler_state = ?\n               AND (jobs.scheduled_for IS NULL OR jobs.scheduled_for <= ?)\n             ORDER BY jobs.queue_position ASC\n             LIMIT ?",
        )
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .bind(Self::current_timestamp())
        .bind(RUNNABLE_JOB_BATCH_SIZE)
        .fetch_all(self.pool())
        .await
        .map_err(Into::into)
    }

    /// Atomically begin context preparation for a queued runnable inference job.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Queued operation whose reducer is entering context preparation.
    pub async fn begin_context_preparation(&self, operation_id: OperationId) -> Result<bool> {
        let now = Self::current_timestamp();
        let result = sqlx::query(
            "UPDATE inference_runs SET run_state = ?, preparing_started_at = ?, last_state_changed_at = ? WHERE operation_id = ? AND run_state = ? AND EXISTS (SELECT 1 FROM agent_jobs WHERE agent_jobs.operation_id = inference_runs.operation_id AND scheduler_state = ?)",
        )
        .bind(InferenceRunStateKind::PreparingContext.to_string())
        .bind(&now)
        .bind(now)
        .bind(operation_id.to_string())
        .bind(InferenceRunStateKind::Queued.to_string())
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .execute(self.pool())
        .await?;

        Ok(result.rows_affected() == 1)
    }

    /// Atomically lease a node and begin waiting for one loaded model to be evicted.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Runnable inference operation that requires the eviction.
    /// * `node_peer_id` - Node selected and exclusively leased for the operation.
    /// * `model_id` - Target model to load after eviction completes.
    /// * `unloading_model_id` - Loaded model selected for eviction.
    /// * `wait_deadline` - Deadline after which the outstanding unload is considered timed out.
    pub async fn begin_model_unloading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        unloading_model_id: ModelId,
        wait_deadline: &str,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let lease = sqlx::query(
            "INSERT INTO node_job_leases (node_client_id, node_device_id, operation_id, acquired_at) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(operation_id.to_string())
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        let run = sqlx::query(
            "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, unloading_model_id = ?, node_selected_at = COALESCE(node_selected_at, ?), last_state_changed_at = ? WHERE operation_id = ? AND run_state = ?",
        )
        .bind(InferenceRunStateKind::UnloadingModel.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(String::from(unloading_model_id))
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(InferenceRunStateKind::PreparingContext.to_string())
        .execute(&mut *tx)
        .await?;

        let job = Self::set_job_waiting(&mut tx, operation_id, wait_deadline, &now).await?;
        if lease.rows_affected() != 1 || run.rows_affected() != 1 || !job {
            tx.rollback().await?;
            return Ok(false);
        }

        sqlx::query(
            "UPDATE nodes SET node_state = ?, last_state_changed_at = ? WHERE client_id = ? AND device_id = ?",
        )
        .bind(nexo_core::NodeStateKind::UnloadingModel.to_string())
        .bind(&now)
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(true)
    }

    /// Atomically lease a node and begin waiting for a model load.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Runnable inference operation that requires model loading.
    /// * `node_peer_id` - Empty node selected and exclusively leased for the operation.
    /// * `model_id` - Target model to load.
    /// * `wait_deadline` - Deadline after which the outstanding load is considered timed out.
    pub async fn begin_model_loading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        wait_deadline: &str,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let lease = sqlx::query(
            "INSERT INTO node_job_leases (node_client_id, node_device_id, operation_id, acquired_at) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(operation_id.to_string())
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        if lease.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }

        let run = sqlx::query(
            "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, node_selected_at = COALESCE(node_selected_at, ?), model_loading_started_at = ?, last_state_changed_at = ? WHERE operation_id = ? AND run_state = ?",
        )
        .bind(InferenceRunStateKind::LoadingModel.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(InferenceRunStateKind::PreparingContext.to_string())
        .execute(&mut *tx)
        .await?;

        let job = sqlx::query(
            "UPDATE agent_jobs SET scheduler_state = ?, waiting_since = ?, wait_deadline = ?, updated_at = ? WHERE operation_id = ? AND scheduler_state = ?",
        )
        .bind(AgentJobSchedulerState::Waiting.to_string())
        .bind(&now)
        .bind(wait_deadline)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .execute(&mut *tx)
        .await?;

        if run.rows_affected() != 1 || job.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }

        tx.commit().await?;
        Ok(true)
    }

    /// Atomically issue the target-model loading phase after eviction was observed complete.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Runnable inference operation resuming after eviction.
    /// * `node_peer_id` - Node already leased by the operation.
    /// * `model_id` - Target model to load.
    /// * `unloading_model_id` - Model whose completed eviction is recorded in the run state.
    /// * `wait_deadline` - Deadline after which the outstanding load is considered timed out.
    pub async fn begin_model_loading_after_unload(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        unloading_model_id: ModelId,
        wait_deadline: &str,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let run = sqlx::query(
            "UPDATE inference_runs SET run_state = ?, unloading_model_id = NULL, model_loading_started_at = ?, last_state_changed_at = ? WHERE operation_id = ? AND run_state = ? AND node_client_id = ? AND node_device_id = ? AND model_id = ? AND unloading_model_id = ? AND EXISTS (SELECT 1 FROM node_job_leases WHERE node_job_leases.operation_id = inference_runs.operation_id AND node_client_id = ? AND node_device_id = ?)",
        )
        .bind(InferenceRunStateKind::LoadingModel.to_string())
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(InferenceRunStateKind::UnloadingModel.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(String::from(unloading_model_id))
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        let job = Self::set_job_waiting(&mut tx, operation_id, wait_deadline, &now).await?;
        if run.rows_affected() != 1 || !job {
            tx.rollback().await?;
            return Ok(false);
        }

        sqlx::query(
            "UPDATE nodes SET node_state = ?, last_state_changed_at = ? WHERE client_id = ? AND device_id = ?",
        )
        .bind(nexo_core::NodeStateKind::LoadingModel.to_string())
        .bind(&now)
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(true)
    }

    /// Atomically record a matching model eviction and wake the job for its next queue tick.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Operation waiting for the eviction.
    /// * `node_peer_id` - Authenticated node that reported completion.
    /// * `unloading_model_id` - Model the node reported as evicted.
    pub async fn complete_model_unloading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        unloading_model_id: ModelId,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let job = sqlx::query(
            "UPDATE agent_jobs SET scheduler_state = ?, waiting_since = NULL, wait_deadline = NULL, updated_at = ? WHERE operation_id = ? AND scheduler_state = ? AND EXISTS (SELECT 1 FROM inference_runs WHERE inference_runs.operation_id = agent_jobs.operation_id AND run_state = ? AND node_client_id = ? AND node_device_id = ? AND unloading_model_id = ?) AND EXISTS (SELECT 1 FROM node_job_leases WHERE node_job_leases.operation_id = agent_jobs.operation_id AND node_client_id = ? AND node_device_id = ?)",
        )
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(AgentJobSchedulerState::Waiting.to_string())
        .bind(InferenceRunStateKind::UnloadingModel.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(unloading_model_id))
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        if job.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }

        sqlx::query(
            "DELETE FROM node_models_in_memory WHERE node_client_id = ? AND node_device_id = ? AND model_id = ?",
        )
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(unloading_model_id))
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE nodes SET node_state = ?, last_state_changed_at = ? WHERE client_id = ? AND device_id = ?",
        )
        .bind(nexo_core::NodeStateKind::Idle.to_string())
        .bind(&now)
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(true)
    }

    /// Move one runnable job into a bounded wait inside an existing transaction.
    ///
    /// # Arguments
    ///
    /// * `tx` - Transaction containing the associated workflow and lease transition.
    /// * `operation_id` - Operation whose scheduler row must currently be runnable.
    /// * `wait_deadline` - Deadline for the external action being initiated.
    /// * `now` - Shared transition timestamp used by all rows in the transaction.
    async fn set_job_waiting(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        operation_id: OperationId,
        wait_deadline: &str,
        now: &str,
    ) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE agent_jobs SET scheduler_state = ?, waiting_since = ?, wait_deadline = ?, updated_at = ? WHERE operation_id = ? AND scheduler_state = ?",
        )
        .bind(AgentJobSchedulerState::Waiting.to_string())
        .bind(now)
        .bind(wait_deadline)
        .bind(now)
        .bind(operation_id.to_string())
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .execute(&mut **tx)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    /// Atomically lease a node and begin inference with an already-loaded model.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Runnable inference operation to dispatch.
    /// * `node_peer_id` - Node with the target model already loaded.
    /// * `model_id` - Loaded model selected for inference.
    /// * `wait_deadline` - Deadline for the outstanding inference operation.
    pub async fn begin_inference_on_loaded_model(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        wait_deadline: &str,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let lease = sqlx::query(
            "INSERT INTO node_job_leases (node_client_id, node_device_id, operation_id, acquired_at) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(operation_id.to_string())
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        if lease.rows_affected() != 1
            || !Self::transition_to_inference(
                &mut tx,
                operation_id,
                node_peer_id,
                model_id,
                InferenceRunStateKind::PreparingContext,
                wait_deadline,
                &now,
            )
            .await?
        {
            tx.rollback().await?;
            return Ok(false);
        }

        tx.commit().await?;
        Ok(true)
    }

    /// Atomically begin inference after a matching model load completed.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Runnable inference operation resuming after model loading.
    /// * `node_peer_id` - Node leased by the operation.
    /// * `model_id` - Model confirmed loaded on the node.
    /// * `wait_deadline` - Deadline for the outstanding inference operation.
    pub async fn begin_inference_after_model_load(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        wait_deadline: &str,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        if !Self::transition_to_inference(
            &mut tx,
            operation_id,
            node_peer_id,
            model_id,
            InferenceRunStateKind::LoadingModel,
            wait_deadline,
            &now,
        )
        .await?
        {
            tx.rollback().await?;
            return Ok(false);
        }

        tx.commit().await?;
        Ok(true)
    }

    async fn transition_to_inference(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
        expected_state: InferenceRunStateKind,
        wait_deadline: &str,
        now: &str,
    ) -> Result<bool> {
        let run = sqlx::query(
            "UPDATE inference_runs SET run_state = ?, node_client_id = ?, node_device_id = ?, model_id = ?, node_selected_at = COALESCE(node_selected_at, ?), in_progress_at = ?, last_state_changed_at = ? WHERE operation_id = ? AND run_state = ? AND (node_client_id IS NULL OR (node_client_id = ? AND node_device_id = ? AND model_id = ?)) AND EXISTS (SELECT 1 FROM node_job_leases WHERE node_job_leases.operation_id = inference_runs.operation_id AND node_client_id = ? AND node_device_id = ?)",
        )
        .bind(InferenceRunStateKind::InProgress.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(expected_state.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut **tx)
        .await?;

        let job = sqlx::query(
            "UPDATE agent_jobs SET scheduler_state = ?, waiting_since = ?, wait_deadline = ?, updated_at = ? WHERE operation_id = ? AND scheduler_state = ?",
        )
        .bind(AgentJobSchedulerState::Waiting.to_string())
        .bind(&now)
        .bind(wait_deadline)
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .execute(&mut **tx)
        .await?;

        Ok(run.rows_affected() == 1 && job.rows_affected() == 1)
    }

    /// Atomically record a matching model-load completion and wake its job.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Operation waiting for model loading.
    /// * `node_peer_id` - Authenticated node that reported completion.
    /// * `model_id` - Model the node reported as loaded.
    pub async fn complete_model_loading(
        &self,
        operation_id: OperationId,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> Result<bool> {
        let now = Self::current_timestamp();
        let mut tx = self.pool().begin().await?;
        let job = sqlx::query(
            "UPDATE agent_jobs SET scheduler_state = ?, waiting_since = NULL, wait_deadline = NULL, updated_at = ? WHERE operation_id = ? AND scheduler_state = ? AND EXISTS (SELECT 1 FROM inference_runs WHERE inference_runs.operation_id = agent_jobs.operation_id AND run_state = ? AND node_client_id = ? AND node_device_id = ? AND model_id = ?) AND EXISTS (SELECT 1 FROM node_job_leases WHERE node_job_leases.operation_id = agent_jobs.operation_id AND node_client_id = ? AND node_device_id = ?)",
        )
        .bind(AgentJobSchedulerState::Runnable.to_string())
        .bind(&now)
        .bind(operation_id.to_string())
        .bind(AgentJobSchedulerState::Waiting.to_string())
        .bind(InferenceRunStateKind::LoadingModel.to_string())
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        if job.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }

        sqlx::query(
            "INSERT INTO node_models_in_memory (node_client_id, node_device_id, model_id) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .bind(String::from(model_id))
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE nodes SET node_state = ?, last_state_changed_at = ? WHERE client_id = ? AND device_id = ?",
        )
        .bind(nexo_core::NodeStateKind::Idle.to_string())
        .bind(now)
        .bind(node_peer_id.client_id().to_string())
        .bind(node_peer_id.device_id().to_string())
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{
        ClientInfo, DeviceInfo, Node, NodeProperties, NodeState, OperationId, User, UserProperties,
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

    fn test_user(name: &str) -> User {
        let properties = UserProperties::new(ClientInfo::new(name), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_node(name: &str) -> Node {
        let properties = NodeProperties::new(ClientInfo::new(name), DeviceInfo::default(), "token");
        Node::from_properties(&properties, NodeState::Idle, HashSet::new())
    }

    #[tokio::test]
    async fn queue_claims_jobs_in_fifo_order() {
        let db = test_db().await;
        let user = test_user("test-user");
        db.connect_user(&user).await.unwrap();

        let first = OperationId::new();
        let second = OperationId::new();
        let first_intent = test_intent(first);
        let second_intent = test_intent(second);
        db.enqueue_inference_job(user.id(), &first_intent)
            .await
            .unwrap();
        db.enqueue_inference_job(user.id(), &second_intent)
            .await
            .unwrap();

        let runnable = db.list_runnable_jobs().await.unwrap();

        assert_eq!(runnable.len(), 2);
        assert_eq!(runnable[0].operation_id, first);
        assert_eq!(runnable[0].queue_position, 1);
        assert_eq!(runnable[0].kind, AgentJobKind::RunInference);
        assert_eq!(runnable[0].user_peer_id, user.id());
    }

    #[tokio::test]
    async fn inference_transitions_enforce_exclusive_node_leases() {
        let db = test_db().await;
        let user = test_user("test-user");
        let node = test_node("test-node");
        db.connect_user(&user).await.unwrap();
        db.connect_node(&node).await.unwrap();

        let first = test_intent(OperationId::new());
        let second = test_intent(OperationId::new());
        db.enqueue_inference_job(user.id(), &first).await.unwrap();
        db.enqueue_inference_job(user.id(), &second).await.unwrap();

        assert!(
            db.begin_context_preparation(first.operation_id)
                .await
                .unwrap()
        );
        assert!(
            db.begin_context_preparation(second.operation_id)
                .await
                .unwrap()
        );
        assert!(
            db.begin_model_loading(
                first.operation_id,
                node.id(),
                ModelId::Gemma4E4bItUqffAfq6,
                "2099-01-01T00:00:00Z",
            )
            .await
            .unwrap()
        );
        assert!(
            !db.begin_model_loading(
                second.operation_id,
                node.id(),
                ModelId::Gemma4E4bItUqffAfq6,
                "2099-01-01T00:00:00Z",
            )
            .await
            .unwrap()
        );
    }

    fn test_intent(operation_id: nexo_core::OperationId) -> nexo_core::InferenceIntent {
        use nexo_core::inference::requests::multimodal::MultiModalPayload;
        use nexo_core::{
            ConversationMessage, InferenceOperation, ModelCapability, ModelSelection,
            ReasoningSettings, SessionId, ToolChoice,
        };

        nexo_core::InferenceIntent {
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
}
