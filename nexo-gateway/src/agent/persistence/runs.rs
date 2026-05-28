//! Run, round, and tool-trace persistence helpers.

use super::db_types::{RoundStatus, RunSummaryKind, ToolTraceStatus, run_status_to_db};
use nexo_core::ReasoningSettings;
use nexo_ws_schema::{Frame, RunStatus};
use sqlx::SqlitePool;

/// Create a new run record.
pub async fn create_run(
    pool: &SqlitePool,
    run_id: &str,
    session_id: &str,
    idempotency_key: &str,
    model_id: Option<&str>,
    reasoning: &ReasoningSettings,
) -> Result<(), sqlx::Error> {
    let reasoning_json = encode_reasoning_json(reasoning)?;
    sqlx::query(
        "INSERT INTO runs (id, session_id, idempotency_key, model_id, reasoning)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(run_id)
    .bind(session_id)
    .bind(idempotency_key)
    .bind(model_id)
    .bind(reasoning_json)
    .execute(pool)
    .await?;
    Ok(())
}

/// Decode serialized reasoning settings read from SQLite.
pub(crate) fn decode_reasoning_json(
    reasoning_json: &str,
) -> Result<ReasoningSettings, sqlx::Error> {
    serde_json::from_str(reasoning_json).map_err(|error| sqlx::Error::Decode(Box::new(error)))
}

fn encode_reasoning_json(reasoning: &ReasoningSettings) -> Result<String, sqlx::Error> {
    serde_json::to_string(reasoning).map_err(|error| sqlx::Error::Encode(Box::new(error)))
}

/// Return the next round index that should be used for a run.
pub async fn next_round_index(pool: &SqlitePool, run_id: &str) -> Result<usize, sqlx::Error> {
    let (next_index,): (i64,) =
        sqlx::query_as("SELECT COALESCE(MAX(round_index), 0) + 1 FROM run_rounds WHERE run_id = ?")
            .bind(run_id)
            .fetch_one(pool)
            .await?;

    Ok(next_index.max(1) as usize)
}

/// Create a new round record for a run and return the round ID.
pub async fn create_round(
    pool: &SqlitePool,
    run_id: &str,
    round_index: usize,
    model_id: Option<&str>,
) -> Result<String, sqlx::Error> {
    let round_id = Frame::new_id();
    sqlx::query(
        "INSERT INTO run_rounds (id, run_id, round_index, status, model_id) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&round_id)
    .bind(run_id)
    .bind(round_index as i64)
    .bind(RoundStatus::Started.as_str())
    .bind(model_id)
    .execute(pool)
    .await?;
    Ok(round_id)
}

/// Mark a round as finished with status, rationale, and selected peer information.
pub async fn finish_round(
    pool: &SqlitePool,
    round_id: &str,
    status: RoundStatus,
    rationale: Option<&str>,
    selected_peer_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE run_rounds
         SET status = ?, rationale = ?, selected_peer_id = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status.as_str())
    .bind(rationale)
    .bind(selected_peer_id)
    .bind(round_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Create a tool trace entry for a round and return its identifier.
pub async fn create_tool_trace(
    pool: &SqlitePool,
    run_id: &str,
    round_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Result<String, sqlx::Error> {
    let trace_id = Frame::new_id();
    let arguments_json = serde_json::to_string(arguments).unwrap_or_default();
    sqlx::query(
        "INSERT INTO tool_traces (id, run_id, round_id, tool_call_id, tool_name, arguments, status)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&trace_id)
    .bind(run_id)
    .bind(round_id)
    .bind(tool_call_id)
    .bind(tool_name)
    .bind(arguments_json)
    .bind(ToolTraceStatus::Started.as_str())
    .execute(pool)
    .await?;
    Ok(trace_id)
}

/// Finish a tool trace with output or error information.
pub async fn finish_tool_trace(
    pool: &SqlitePool,
    trace_id: &str,
    status: ToolTraceStatus,
    output: Option<&str>,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE tool_traces
         SET status = ?, output = ?, error = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status.as_str())
    .bind(output)
    .bind(error)
    .bind(trace_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Store a summary record for a run.
pub async fn store_run_summary(
    pool: &SqlitePool,
    run_id: &str,
    round_id: Option<&str>,
    kind: RunSummaryKind,
    content: &str,
) -> Result<String, sqlx::Error> {
    let summary_id = Frame::new_id();
    sqlx::query(
        "INSERT INTO run_summaries (id, run_id, round_id, kind, content)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&summary_id)
    .bind(run_id)
    .bind(round_id)
    .bind(kind.as_str())
    .bind(content)
    .execute(pool)
    .await?;
    Ok(summary_id)
}

/// Stop an active run and return its session ID when the stop was applied.
pub async fn stop_run(pool: &SqlitePool, run_id: &str) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let result = sqlx::query(
        "UPDATE runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(run_status_to_db(RunStatus::Cancelled))
    .bind(run_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        return Ok(None);
    }

    Ok(Some(session_id))
}

/// Return whether a run has already been marked as cancelled.
pub async fn is_run_cancelled(pool: &SqlitePool, run_id: &str) -> Result<bool, sqlx::Error> {
    let row: Option<(String,)> = sqlx::query_as("SELECT status FROM runs WHERE id = ?")
        .bind(run_id)
        .fetch_optional(pool)
        .await?;

    Ok(matches!(
        row.as_ref().map(|(status,)| status.as_str()),
        Some(status) if status == run_status_to_db(RunStatus::Cancelled)
    ))
}

/// Mark a run as queued while it waits for an inference-capable node.
pub async fn mark_run_queued(pool: &SqlitePool, run_id: &str) {
    if let Err(error) = sqlx::query(
        "UPDATE runs SET status = ?, queued_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(run_status_to_db(RunStatus::Queued))
    .bind(run_id)
    .execute(pool)
    .await
    {
        tracing::error!("Failed to queue run {run_id}: {error}");
    } else {
        tracing::info!("Run {run_id} queued (no LLM available)");
    }
}

/// Mark a run as finished with a status and optional summary.
pub async fn finish_run(
    pool: &SqlitePool,
    run_id: &str,
    status: RunStatus,
    summary: Option<&str>,
) -> Result<(), sqlx::Error> {
    let result = sqlx::query(
        "UPDATE runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(run_status_to_db(status))
    .bind(run_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 1
        && let Some(summary_text) = summary
    {
        let kind = RunSummaryKind::from_terminal_run_status(status);
        let _ = store_run_summary(pool, run_id, None, kind, summary_text).await;
    }

    Ok(())
}
