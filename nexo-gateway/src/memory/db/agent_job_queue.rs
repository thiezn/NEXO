use super::DbClient;
use super::db_types::AgentJobQueueRecord;
use crate::Result;
use crate::agent::{AgentJobKind, AgentJobQueueStatus};
use nexo_core::{OperationId, PeerId};

impl DbClient {
    /// Enqueue a new inference job at the back of the FIFO queue.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier represented by the queued job.
    /// * `user_peer_id` - The user peer that owns the queued operation.
    pub async fn enqueue_inference_job(
        &self,
        operation_id: OperationId,
        user_peer_id: PeerId,
    ) -> Result<i64> {
        let enqueued_at = Self::current_timestamp();
        let result = sqlx::query(
            "INSERT INTO agent_job_queue (operation_id, user_client_id, user_device_id, job_kind, status, enqueued_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(operation_id.to_string())
        .bind(user_peer_id.client_id().to_string())
        .bind(user_peer_id.device_id().to_string())
        .bind(AgentJobKind::RunInference.to_string())
        .bind(AgentJobQueueStatus::Queued.to_string())
        .bind(enqueued_at)
        .execute(self.pool())
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Claim the next queued job in FIFO order.
    ///
    /// This method stays crate-local because it returns an internal DB projection.
    async fn claim_next_job(&self) -> Result<Option<AgentJobQueueRecord>> {
        let mut tx = self.pool().begin().await?;
        let row = sqlx::query_as::<_, AgentJobQueueRecord>(
            "SELECT queue_position, operation_id, user_client_id, user_device_id, job_kind, status, attempt_count, failure_message\n             FROM agent_job_queue\n             WHERE status = ?\n             ORDER BY queue_position ASC\n             LIMIT 1",
        )
        .bind(AgentJobQueueStatus::Queued.to_string())
        .fetch_optional(&mut *tx)
        .await?;

        let Some(mut record) = row else {
            tx.commit().await?;
            return Ok(None);
        };

        let claimed_at = Self::current_timestamp();

        sqlx::query(
            "UPDATE agent_job_queue SET status = ?, attempt_count = attempt_count + 1, claimed_at = ? WHERE queue_position = ?",
        )
        .bind(AgentJobQueueStatus::Claimed.to_string())
        .bind(claimed_at)
        .bind(record.queue_position)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        record.status = AgentJobQueueStatus::Claimed;
        record.attempt_count += 1;

        Ok(Some(record))
    }

    /// Mark a claimed job as completed.
    ///
    /// # Arguments
    ///
    /// * `queue_position` - The FIFO row identifier returned when the job was enqueued.
    pub async fn complete_job(&self, queue_position: i64) -> Result {
        self.finish_job(queue_position, AgentJobQueueStatus::Completed, None).await
    }

    /// Mark a claimed job as failed with a message.
    ///
    /// # Arguments
    ///
    /// * `queue_position` - The FIFO row identifier returned when the job was enqueued.
    /// * `failure_message` - The error text to persist for later inspection.
    pub async fn fail_job(&self, queue_position: i64, failure_message: &str) -> Result {
        self.finish_job(queue_position, AgentJobQueueStatus::Failed, Some(failure_message)).await
    }

    /// Finalize a claimed queue row with a terminal status.
    ///
    /// # Arguments
    ///
    /// * `queue_position` - The FIFO row identifier to update.
    /// * `status` - The terminal status to persist for the queue row.
    /// * `failure_message` - The optional failure text to store for failed jobs.
    async fn finish_job(
        &self,
        queue_position: i64,
        status: AgentJobQueueStatus,
        failure_message: Option<&str>,
    ) -> Result {
        let finished_at = Self::current_timestamp();
        sqlx::query(
            "UPDATE agent_job_queue SET status = ?, failure_message = ?, finished_at = ? WHERE queue_position = ?",
        )
        .bind(status.to_string())
        .bind(failure_message)
        .bind(finished_at)
        .bind(queue_position)
        .execute(self.pool())
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, OperationId, User, UserProperties};
    use sqlx::sqlite::SqlitePoolOptions;

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

    #[tokio::test]
    async fn queue_claims_jobs_in_fifo_order() {
        let db = test_db().await;
        let user = test_user("test-user");
        db.connect_user(&user).await.unwrap();

        let first = OperationId::new();
        let second = OperationId::new();
        db.create_operation(first, user.id()).await.unwrap();
        db.create_operation(second, user.id()).await.unwrap();
        db.enqueue_inference_job(first, user.id()).await.unwrap();
        db.enqueue_inference_job(second, user.id()).await.unwrap();

        let claimed_first = db.claim_next_job().await.unwrap().unwrap();
        let claimed_second = db.claim_next_job().await.unwrap().unwrap();

        assert_eq!(claimed_first.operation_id, first);
        assert_eq!(claimed_second.operation_id, second);
    }
}
