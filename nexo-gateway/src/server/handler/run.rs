//! WebSocket handlers for run and session lifecycle requests.

use crate::agent::{RunCommand, RunHandle};
use crate::server::state::SharedState;
use nexo_ws_schema::{
    ErrorPayload, EventKind, Frame, RunEventPayload, RunInstructionsAppendParams,
    RunInstructionsAppendResponse, RunStartParams, RunStatus, RunStopParams, RunStopResponse,
    SessionClearParams, SessionCreateParams, SessionGetParams,
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
            match crate::agent::session::create_session(db, &user_id, None, None::<&str>).await {
                Ok(pair) => pair,
                Err(e) => {
                    return internal_error(request_id, format!("Failed to create session: {e}"));
                }
            }
        }
    };

    let run_id = Frame::new_id();
    let thinking = run_params.thinking.unwrap_or(false);
    if let Err(e) = crate::agent::session::create_run(
        db,
        &run_id,
        &session_id,
        &run_params.idempotency_key,
        run_params.model_id.as_deref(),
        thinking,
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
        peer_id: peer_id.to_string(),
        model_id: run_params.model_id,
        prompt_collection_id,
        thinking,
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

    let stopped_session = match crate::agent::session::stop_run(db, &stop_params.run_id).await {
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

    let message_id = match crate::agent::session::append_run_context(
        db,
        &append_params.run_id,
        &append_params.instructions,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return internal_error(request_id, format!("Failed to append instructions: {error}"));
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

/// Handle `session.create` requests.
pub(super) async fn handle_session_create(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let session_params: SessionCreateParams =
        match parse_params(request_id, params, "session.create") {
            Ok(p) => p,
            Err(f) => return f,
        };
    let user_id = resolve_user_id(state, peer_id).await;

    match crate::agent::session::create_session(
        db,
        &user_id,
        session_params.name.as_deref(),
        session_params.prompt_collection_id.as_deref(),
    )
    .await
    {
        Ok((session_id, prompt_collection_id)) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::SessionCreateResponse {
                session_id,
                prompt_collection_id,
            },
        ),
        Err(e) => internal_error(request_id, format!("Failed to create session: {e}")),
    }
}

/// Handle `session.list` requests for the calling user.
pub(super) async fn handle_session_list(
    request_id: &str,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let user_id = resolve_user_id(state, peer_id).await;

    match crate::agent::session::list_sessions(db, &user_id).await {
        Ok(sessions) => {
            ok_or_internal_error(request_id, nexo_ws_schema::SessionListResponse { sessions })
        }
        Err(e) => internal_error(request_id, format!("Failed to list sessions: {e}")),
    }
}

/// Handle `session.get` requests.
pub(super) async fn handle_session_get(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let get_params: SessionGetParams = match parse_params(request_id, params, "session.get") {
        Ok(p) => p,
        Err(f) => return f,
    };

    match crate::agent::session::get_session(db, &get_params.session_id).await {
        Ok(Some(resp)) => ok_or_internal_error(request_id, resp),
        Ok(None) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "session_not_found".into(),
                message: format!("Session '{}' not found", get_params.session_id),
            },
        ),
        Err(e) => internal_error(request_id, format!("Failed to get session: {e}")),
    }
}

/// Handle `session.clear` requests.
pub(super) async fn handle_session_clear(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let clear_params: SessionClearParams = match parse_params(request_id, params, "session.clear") {
        Ok(p) => p,
        Err(f) => return f,
    };

    match crate::agent::session::clear_session(db, &clear_params.session_id).await {
        Ok(cleared) => {
            ok_or_internal_error(request_id, nexo_ws_schema::SessionClearResponse { cleared })
        }
        Err(e) => internal_error(request_id, format!("Failed to clear session: {e}")),
    }
}
