//! WebSocket handlers for run lifecycle requests.

use crate::agent::{RunCommand, RunHandle};
use crate::server::state::SharedState;
use nexo_ws_schema::{
    EventKind, Frame, RunEventPayload, RunInstructionsAppendParams, RunInstructionsAppendResponse,
    RunStartParams, RunStatus, RunStopParams, RunStopResponse,
};
use sqlx::SqlitePool;

use super::base::{internal_error, ok_or_internal_error, parse_params, resolve_user_id};

/// Handle `run.start` requests by creating a run and submitting it to the background task.
pub(super) async fn handle_run_start(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    run_handle: &RunHandle,
) -> Frame {
    let run_params: RunStartParams = match parse_params(request_id, params, "run.start") {
        Ok(p) => p,
        Err(f) => return f,
    };

    let user_id = resolve_user_id(state, peer_id).await;

    let (session_id, prompt_collection_id) = match run_params.session_id {
        Some(sid) => {
            let pcid: Option<String> = sqlx::query_as::<_, (Option<String>,)>(
                "SELECT prompt_collection_id FROM sessions WHERE id = ?",
            )
            .bind(&sid)
            .fetch_optional(db)
            .await
            .ok()
            .flatten()
            .and_then(|(c,)| c);
            (sid, pcid)
        }
        None => {
            match crate::agent::persistence::create_session(db, &user_id, None, None::<&str>).await
            {
                Ok(pair) => pair,
                Err(e) => {
                    return internal_error(request_id, format!("Failed to create session: {e}"));
                }
            }
        }
    };

    let run_id = Frame::new_id();
    let reasoning = run_params.reasoning.unwrap_or_default();
    if let Err(e) = crate::agent::persistence::create_run(
        db,
        &run_id,
        &session_id,
        &run_params.idempotency_key,
        run_params.model_id.as_deref(),
        &reasoning,
    )
    .await
    {
        return internal_error(request_id, format!("Failed to create run: {e}"));
    }

    let cmd = RunCommand::StartRun {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        input: run_params.input,
        instructions: run_params.instructions,
        model_id: run_params.model_id,
        prompt_collection_id,
        reasoning,
    };
    if let Err(e) = run_handle.submit(cmd).await {
        tracing::error!("Failed to submit run command: {e}");
    }

    ok_or_internal_error(
        request_id,
        nexo_ws_schema::RunStartResponse {
            run_id,
            session_id,
            status: RunStatus::Accepted,
            summary: None,
        },
    )
}

/// Handle `run.stop` requests by marking the run as cancelled.
pub(super) async fn handle_run_stop(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let stop_params: RunStopParams = match parse_params(request_id, params, "run.stop") {
        Ok(params) => params,
        Err(frame) => return frame,
    };

    let stopped_session = match crate::agent::persistence::stop_run(db, &stop_params.run_id).await {
        Ok(result) => result,
        Err(error) => {
            return internal_error(request_id, format!("Failed to stop run: {error}"));
        }
    };

    let stopped = stopped_session.is_some();

    if let Some(session_id) = stopped_session.as_ref() {
        let event = Frame::event(
            EventKind::Run,
            RunEventPayload {
                run_id: stop_params.run_id.clone(),
                session_id: session_id.clone(),
                status: RunStatus::Cancelled,
                content: None,
                tool_name: None,
                tool_call_id: None,
                error: None,
                thinking_content: None,
            },
        );
        if let Ok(frame) = event {
            let sender = state.read().await.event_tx.clone();
            let _ = sender.send(frame);
        }
    }

    ok_or_internal_error(request_id, RunStopResponse { stopped })
}

/// Handle `run.instructions.append` requests for active runs.
pub(super) async fn handle_run_instructions_append(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let append_params: RunInstructionsAppendParams =
        match parse_params(request_id, params, "run.instructions.append") {
            Ok(params) => params,
            Err(frame) => return frame,
        };

    let message_id = match crate::agent::persistence::append_run_instructions(
        db,
        &append_params.run_id,
        &append_params.instructions,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return internal_error(
                request_id,
                format!("Failed to append instructions: {error}"),
            );
        }
    };

    ok_or_internal_error(
        request_id,
        RunInstructionsAppendResponse {
            queued: message_id.is_some(),
            message_id,
        },
    )
}
