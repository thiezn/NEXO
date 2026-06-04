//! WebSocket handlers for session lifecycle requests.

use crate::server::state::SharedState;
use nexo_ws_schema::{
    ErrorPayload, EventKind, Frame, SessionClearParams, SessionClosedPayload, SessionCreateParams,
    SessionGetParams,
};
use sqlx::SqlitePool;

use super::base::{internal_error, ok_or_internal_error, parse_params, resolve_user_id};

/// Handle `session.create` requests.
pub(super) async fn handle_create(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let session_params: SessionCreateParams =
        match parse_params(request_id, params, "session.create") {
            Ok(params) => params,
            Err(frame) => return frame,
        };
    let user_id = resolve_user_id(state, peer_id).await;

    match crate::agent::persistence::create_session(
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
        Err(error) => internal_error(request_id, format!("Failed to create session: {error}")),
    }
}

/// Handle `session.list` requests for the calling user.
pub(super) async fn handle_list(
    request_id: &str,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let user_id = resolve_user_id(state, peer_id).await;

    match crate::agent::persistence::list_sessions(db, &user_id).await {
        Ok(sessions) => {
            ok_or_internal_error(request_id, nexo_ws_schema::SessionListResponse { sessions })
        }
        Err(error) => internal_error(request_id, format!("Failed to list sessions: {error}")),
    }
}

/// Handle `session.get` requests.
pub(super) async fn handle_get(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let get_params: SessionGetParams = match parse_params(request_id, params, "session.get") {
        Ok(params) => params,
        Err(frame) => return frame,
    };

    match crate::agent::persistence::get_session(db, &get_params.session_id).await {
        Ok(Some(response)) => ok_or_internal_error(request_id, response),
        Ok(None) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "session_not_found".into(),
                message: format!("Session '{}' not found", get_params.session_id),
            },
        ),
        Err(error) => internal_error(request_id, format!("Failed to get session: {error}")),
    }
}

/// Handle `session.clear` requests.
pub(super) async fn handle_clear(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
    db: &SqlitePool,
) -> Frame {
    let clear_params: SessionClearParams = match parse_params(request_id, params, "session.clear") {
        Ok(params) => params,
        Err(frame) => return frame,
    };

    match crate::agent::persistence::clear_session(db, &clear_params.session_id).await {
        Ok(cleared) => {
            if cleared {
                let event = Frame::event(
                    EventKind::SessionClosed,
                    SessionClosedPayload {
                        session_id: clear_params.session_id.clone(),
                    },
                );
                if let Ok(event) = event {
                    let state = state.read().await;
                    let _ = state.event_tx.send(event);
                }
            }
            ok_or_internal_error(request_id, nexo_ws_schema::SessionClearResponse { cleared })
        }
        Err(error) => internal_error(request_id, format!("Failed to clear session: {error}")),
    }
}
