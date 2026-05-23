//! Run, round, and tool-trace persistence helpers.

use nexo_ws_schema::{AgentStatus, Frame};
use sqlx::SqlitePool;

/// Create a new agent run record.
pub async fn create_run(
    pool: &SqlitePool,
    run_id: &str,
    session_id: &str,
    idempotency_key: &str,
    model_id: Option<&str>,
    thinking: bool,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO agent_runs (id, session_id, idempotency_key, model_id, thinking)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(run_id)
    .bind(session_id)
    .bind(idempotency_key)
    .bind(model_id)
    .bind(if thinking { 1 } else { 0 })
    .execute(pool)
    .await?;
    Ok(())
}

/// Return the next round index that should be used for a run.
pub async fn next_round_index(pool: &SqlitePool, run_id: &str) -> Result<usize, sqlx::Error> {
    let (next_index,): (i64,) = sqlx::query_as(
        "SELECT COALESCE(MAX(round_index), 0) + 1 FROM agent_rounds WHERE run_id = ?",
    )
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
    sqlx::query("INSERT INTO agent_rounds (id, run_id, round_index, model_id) VALUES (?, ?, ?, ?)")
        .bind(&round_id)
        .bind(run_id)
        .bind(round_index as i64)
        .bind(model_id)
        .execute(pool)
        .await?;
    Ok(round_id)
}

/// Mark a round as finished with status, rationale, and selected peer information.
pub async fn finish_round(
    pool: &SqlitePool,
    round_id: &str,
    status: &str,
    rationale: Option<&str>,
    selected_peer_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE agent_rounds
         SET status = ?, rationale = ?, selected_peer_id = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status)
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
        "INSERT INTO tool_traces (id, run_id, round_id, tool_call_id, tool_name, arguments)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&trace_id)
    .bind(run_id)
    .bind(round_id)
    .bind(tool_call_id)
    .bind(tool_name)
    .bind(arguments_json)
    .execute(pool)
    .await?;
    Ok(trace_id)
}

/// Finish a tool trace with output or error information.
pub async fn finish_tool_trace(
    pool: &SqlitePool,
    trace_id: &str,
    success: bool,
    output: Option<&str>,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    let status = if success { "completed" } else { "failed" };
    sqlx::query(
        "UPDATE tool_traces
         SET status = ?, output = ?, error = ?, finished_at = datetime('now')
         WHERE id = ?",
    )
    .bind(status)
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
    kind: &str,
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
    .bind(kind)
    .bind(content)
    .execute(pool)
    .await?;
    Ok(summary_id)
}

/// Stop an active run and return its session ID when the stop was applied.
pub async fn stop_run(pool: &SqlitePool, run_id: &str) -> Result<Option<String>, sqlx::Error> {
    let session_row: Option<(String,)> =
        sqlx::query_as("SELECT session_id FROM agent_runs WHERE id = ? AND finished_at IS NULL")
            .bind(run_id)
            .fetch_optional(pool)
            .await?;

    let Some((session_id,)) = session_row else {
        return Ok(None);
    };

    let status = serde_json::to_value(AgentStatus::Cancelled)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "cancelled".to_string());
    let result = sqlx::query(
        "UPDATE agent_runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(status)
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
    let row: Option<(String,)> = sqlx::query_as("SELECT status FROM agent_runs WHERE id = ?")
        .bind(run_id)
        .fetch_optional(pool)
        .await?;

    Ok(matches!(
        row.as_ref().map(|(status,)| status.as_str()),
        Some("cancelled")
    ))
}

/// Mark an agent run as finished with a status and optional summary.
pub async fn finish_run(
    pool: &SqlitePool,
    run_id: &str,
    status: AgentStatus,
    summary: Option<&str>,
) -> Result<(), sqlx::Error> {
    let status_str = serde_json::to_value(status)
        .ok()
        .and_then(|value| value.as_str().map(String::from));
    let result = sqlx::query(
        "UPDATE agent_runs SET status = ?, finished_at = datetime('now') \
         WHERE id = ? AND finished_at IS NULL",
    )
    .bind(status_str.as_deref().unwrap_or("failed"))
    .bind(run_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 1
        && let Some(summary_text) = summary
    {
        let kind = match status {
            AgentStatus::Completed => "final_response",
            AgentStatus::Failed => "failure",
            AgentStatus::Cancelled => "cancelled",
            _ => "terminal_state",
        };
        let _ = store_run_summary(pool, run_id, None, kind, summary_text).await;
    }

    Ok(())
}
