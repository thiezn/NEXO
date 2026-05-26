use nexo_ws_schema::{Frame, RunStatus};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

use super::events;

/// Mark a run as queued while it waits for an inference-capable node.
pub async fn mark_run_queued(pool: &SqlitePool, run_id: &str) {
    if let Err(error) = sqlx::query(
        "UPDATE runs SET status = 'queued', queued_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(run_id)
    .execute(pool)
    .await
    {
        tracing::error!("Failed to queue run {run_id}: {error}");
    } else {
        tracing::info!("Run {run_id} queued (no LLM available)");
    }
}
