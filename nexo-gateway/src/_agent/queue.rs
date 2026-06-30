use crate::server::state::SharedState;
use nexo_core::ToolChoice;
use nexo_ws_schema::{Frame, RunStatus};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

/// Drain all queued runs, resuming them in the order they were queued.
///
/// This is called whenever a new LLM-capable node connects. Since the background run task is a
/// task, all DrainQueue commands serialize naturally — there is no double-drain race.
pub async fn drain_queue(
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
) {
    let queued_status = crate::agent::persistence::run_status_to_db(RunStatus::Queued);
    let accepted_status = crate::agent::persistence::run_status_to_db(RunStatus::Accepted);

    // Fetch all queued runs in order they were queued
    let queued: Vec<(String, String, Option<String>, String)> = match sqlx::query_as(
        "SELECT ar.id, ar.session_id, ar.model_id, ar.reasoning
            FROM runs ar
            WHERE ar.status = ?
            ORDER BY ar.queued_at ASC",
    )
    .bind(queued_status)
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
        tracing::info!("No queued runs to drain");
        return;
    }

    tracing::info!("Draining {} queued run(s)", queued.len());

    for (run_id, session_id, model_id, reasoning_json) in queued {
        // Claim the run by setting status to 'accepted'. This prevents re-processing if another
        // DrainQueue is somehow triggered before this one finishes.
        let result =
            sqlx::query("UPDATE runs SET status = ?, queued_at = NULL WHERE id = ? AND status = ?")
                .bind(accepted_status)
                .bind(&run_id)
                .bind(queued_status)
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

        tracing::info!("Resuming queued run {run_id} (session={session_id})");

        // Look up the session's prompt_collection_id for this queued run
        let prompt_collection_id: Option<String> = sqlx::query_as::<_, (Option<String>,)>(
            "SELECT prompt_collection_id FROM sessions WHERE id = ?",
        )
        .bind(&session_id)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
        .and_then(|(c,)| c);

        let reasoning = match crate::agent::persistence::decode_reasoning_json(&reasoning_json) {
            Ok(reasoning) => reasoning,
            Err(error) => {
                tracing::error!("Failed to decode reasoning for queued run {run_id}: {error}");
                continue;
            }
        };
        let tool_choice = match crate::agent::persistence::decode_tool_choice_json(&reasoning_json)
        {
            Ok(tool_choice) => tool_choice,
            Err(error) => {
                tracing::error!("Failed to decode tool choice for queued run {run_id}: {error}");
                ToolChoice::Automatic
            }
        };

        // If the node disappeared between the model.status update and this drain pass,
        // the run loop will cleanly queue the run again via `InferenceOutcome::NoLlmAvailable`.
        super::r#loop::run_existing(
            &run_id,
            &session_id,
            db,
            state,
            event_tx,
            model_id.as_deref(),
            prompt_collection_id.as_deref(),
            reasoning,
            tool_choice,
        )
        .await;
    }
}
