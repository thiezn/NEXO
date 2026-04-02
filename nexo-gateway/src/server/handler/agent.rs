use crate::agent::{AgentCommand, AgentHandle};
use crate::server::state::SharedState;
use nexo_ws_schema::{
    AgentParams, AgentStatus, ErrorPayload, Frame, SessionClearParams, SessionCreateParams,
    SessionGetParams,
};
use sqlx::SqlitePool;

use super::base::{internal_error, ok_or_internal_error, parse_params, resolve_user_id};

pub(super) async fn handle_agent(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    agent_handle: &AgentHandle,
) -> Frame {
    let agent_params: AgentParams = match parse_params(request_id, params, "agent") {
        Ok(p) => p,
        Err(f) => return f,
    };

    let user_id = resolve_user_id(state, peer_id).await;

    let (session_id, prefill_collection_id) = match agent_params.session_id {
        Some(sid) => {
            let pcid: Option<String> =
                sqlx::query_as::<_, (Option<String>,)>(
                    "SELECT prefill_collection_id FROM sessions WHERE id = ?",
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
                    return internal_error(
                        request_id,
                        format!("Failed to create session: {e}"),
                    )
                }
            }
        }
    };

    let run_id = Frame::new_id();
    if let Err(e) = crate::agent::session::create_run(
        db,
        &run_id,
        &session_id,
        &agent_params.idempotency_key,
        agent_params.model_id.as_deref(),
    )
    .await
    {
        return internal_error(request_id, format!("Failed to create run: {e}"));
    }

    let cmd = AgentCommand::RunAgent {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        prompt: agent_params.prompt,
        context: agent_params.context,
        peer_id: peer_id.to_string(),
        model_id: agent_params.model_id,
        prefill_collection_id,
    };
    if let Err(e) = agent_handle.submit(cmd).await {
        tracing::error!("Failed to submit agent command: {e}");
    }

    ok_or_internal_error(
        request_id,
        nexo_ws_schema::AgentResponse {
            run_id,
            session_id,
            status: AgentStatus::Accepted,
            summary: None,
        },
    )
}

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
        session_params.prefill_collection_id.as_deref(),
    )
    .await
    {
        Ok((session_id, prefill_collection_id)) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::SessionCreateResponse {
                session_id,
                prefill_collection_id,
            },
        ),
        Err(e) => internal_error(request_id, format!("Failed to create session: {e}")),
    }
}

pub(super) async fn handle_session_list(
    request_id: &str,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let user_id = resolve_user_id(state, peer_id).await;

    match crate::agent::session::list_sessions(db, &user_id).await {
        Ok(sessions) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::SessionListResponse { sessions },
        ),
        Err(e) => internal_error(request_id, format!("Failed to list sessions: {e}")),
    }
}

pub(super) async fn handle_session_get(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let get_params: SessionGetParams =
        match parse_params(request_id, params, "session.get") {
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

pub(super) async fn handle_session_clear(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let clear_params: SessionClearParams =
        match parse_params(request_id, params, "session.clear") {
            Ok(p) => p,
            Err(f) => return f,
        };

    match crate::agent::session::clear_session(db, &clear_params.session_id).await {
        Ok(cleared) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::SessionClearResponse { cleared },
        ),
        Err(e) => internal_error(request_id, format!("Failed to clear session: {e}")),
    }
}
