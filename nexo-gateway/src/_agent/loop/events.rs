use nexo_ws_schema::{EventKind, Frame, RunEventPayload, RunStatus};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

/// Emit a simple run status event.
pub fn emit_status(
    event_tx: &broadcast::Sender<Frame>,
    run_id: &str,
    session_id: &str,
    status: RunStatus,
    content: Option<&str>,
    tool_name: Option<&str>,
) {
    emit_status_with_thinking(
        event_tx, run_id, session_id, status, content, tool_name, None,
    );
}

/// Emit a run status event that may also carry ephemeral thinking content.
pub fn emit_status_with_thinking(
    event_tx: &broadcast::Sender<Frame>,
    run_id: &str,
    session_id: &str,
    status: RunStatus,
    content: Option<&str>,
    tool_name: Option<&str>,
    thinking_content: Option<&str>,
) {
    let payload = RunEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status,
        content: content.map(str::to_owned),
        tool_name: tool_name.map(str::to_owned),
        tool_call_id: None,
        error: None,
        thinking_content: thinking_content.map(str::to_owned),
    };
    if let Ok(frame) = Frame::event(EventKind::Run, &payload) {
        let _ = event_tx.send(frame);
    }
}

/// Emit the start of a tool call for the active run.
pub fn emit_tool_started(
    event_tx: &broadcast::Sender<Frame>,
    run_id: &str,
    session_id: &str,
    tool_name: &str,
    tool_call_id: &str,
) {
    let payload = RunEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status: RunStatus::ToolCall,
        content: None,
        tool_name: Some(tool_name.to_string()),
        tool_call_id: Some(tool_call_id.to_string()),
        error: None,
        thinking_content: None,
    };
    if let Ok(frame) = Frame::event(EventKind::Run, &payload) {
        let _ = event_tx.send(frame);
    }
}

/// Emit a completed tool result for the active run.
pub fn emit_tool_result(
    event_tx: &broadcast::Sender<Frame>,
    run_id: &str,
    session_id: &str,
    tool_name: &str,
    tool_call_id: &str,
    content: &str,
) {
    let payload = RunEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status: RunStatus::ToolCall,
        content: Some(content.to_string()),
        tool_name: Some(tool_name.to_string()),
        tool_call_id: Some(tool_call_id.to_string()),
        error: None,
        thinking_content: None,
    };
    if let Ok(frame) = Frame::event(EventKind::Run, &payload) {
        let _ = event_tx.send(frame);
    }
}

/// Emit a queued status event for a paused run.
pub fn emit_queued_event(event_tx: &broadcast::Sender<Frame>, run_id: &str, session_id: &str) {
    emit_status_with_thinking(
        event_tx,
        run_id,
        session_id,
        RunStatus::Queued,
        Some(
            "No inference node is currently available. Your request has been queued and will be processed as soon as a node becomes available.",
        ),
        None,
        None,
    );
}

/// Emit a failed terminal event and finish the run in storage.
pub async fn fail_run(
    event_tx: &broadcast::Sender<Frame>,
    pool: &SqlitePool,
    run_id: &str,
    session_id: &str,
    error: &str,
) {
    tracing::error!("Run {run_id} failed: {error}");
    let payload = RunEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status: RunStatus::Failed,
        content: None,
        tool_name: None,
        tool_call_id: None,
        error: Some(error.to_string()),
        thinking_content: None,
    };
    if let Ok(frame) = Frame::event(EventKind::Run, &payload) {
        let _ = event_tx.send(frame);
    }
    let _ =
        crate::agent::persistence::finish_run(pool, run_id, RunStatus::Failed, Some(error)).await;
    crate::agent::locks::release_all_for_run(pool, run_id)
        .await
        .ok();
}
