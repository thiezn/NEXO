use crate::server::state::SharedState;
use nexo_ws_schema::Frame;
use sqlx::SqlitePool;
use tokio::sync::broadcast;

/// Drain all queued agent runs, submitting them to the loop runner in the order they were queued.
///
/// This is called whenever a new LLM-capable node connects. Since `agent_task` is a single async
/// task, all DrainQueue commands serialize naturally — there is no double-drain race.
pub async fn drain_queue(
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
) {
    // Return early if no LLM node is connected
    if !state.read().await.has_llm_peer() {
        return;
    }

    // Fetch all queued runs in order they were queued
    let queued: Vec<(String, String, String, Option<String>, Option<String>, Option<String>)> =
        match sqlx::query_as(
            "SELECT ar.id, ar.session_id, ar.queued_prompt, ar.queued_context,
                    ar.model_id, ar.queued_peer_id
             FROM agent_runs ar
             WHERE ar.status = 'queued'
             ORDER BY ar.queued_at ASC",
        )
        .fetch_all(db)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::error!("Failed to fetch queued runs: {e}");
                return;
            }
        };

    if queued.is_empty() {
        return;
    }

    tracing::info!("Draining {} queued agent run(s)", queued.len());

    for (run_id, session_id, prompt, context_json, model_id, peer_id) in queued {
        // Claim the run by setting status to 'accepted'. This prevents re-processing if another
        // DrainQueue is somehow triggered before this one finishes.
        let result = sqlx::query(
            "UPDATE agent_runs SET status = 'accepted', queued_at = NULL WHERE id = ? AND status = 'queued'",
        )
        .bind(&run_id)
        .execute(db)
        .await;

        match result {
            Ok(r) if r.rows_affected() == 1 => {}
            Ok(_) => {
                // Another drain claimed it already
                tracing::debug!("Run {run_id} already claimed by another drain, skipping");
                continue;
            }
            Err(e) => {
                tracing::error!("Failed to claim queued run {run_id}: {e}");
                continue;
            }
        }

        let context: Option<serde_json::Value> = context_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());

        let peer_id_ref = peer_id.as_deref().unwrap_or("queue-drain");

        tracing::info!("Resuming queued run {run_id} (session={session_id})");

        // Look up the session's prefill_collection_id for this queued run
        let prefill_collection_id: Option<String> = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT prefill_collection_id FROM sessions WHERE id = ?",
        )
        .bind(&session_id)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
        .and_then(|(c,)| c);

        super::loop_runner::run(
            &run_id,
            &session_id,
            &prompt,
            context.as_ref(),
            peer_id_ref,
            db,
            state,
            event_tx,
            model_id.as_deref(),
            prefill_collection_id.as_deref(),
        )
        .await;
    }
}
