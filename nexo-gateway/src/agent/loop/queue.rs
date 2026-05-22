use nexo_ws_schema::{AgentStatus, Frame};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

use super::events;

/// Mark a run as queued while it waits for an inference-capable node.
pub async fn mark_run_queued(pool: &SqlitePool, run_id: &str) {
    if let Err(error) = sqlx::query(
        "UPDATE agent_runs SET status = 'queued', queued_at = datetime('now') \
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

/// Emit a queued status event for a paused run.
pub fn emit_queued_event(event_tx: &broadcast::Sender<Frame>, run_id: &str, session_id: &str) {
    events::emit_status_with_thinking(
        event_tx,
        run_id,
        session_id,
        AgentStatus::Queued,
        Some(
            "No inference node is currently available. Your request has been queued and will be processed as soon as a node becomes available.",
        ),
        None,
        None,
    );
}
