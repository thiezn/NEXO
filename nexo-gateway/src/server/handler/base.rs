use crate::server::state::SharedState;
use nexo_ws_schema::{ErrorPayload, Frame, Method};
use std::time::Duration;
use tokio::sync::mpsc;

pub(super) const IMAGE_ANALYSIS_TIMEOUT: Duration = Duration::from_secs(180);
pub(super) const AUDIO_ANALYSIS_TIMEOUT: Duration = Duration::from_secs(180);

/// Build an ok response, falling back to an internal_error response on serialization failure.
pub(super) fn ok_or_internal_error(request_id: &str, payload: impl serde::Serialize) -> Frame {
    Frame::ok_response(request_id, payload).unwrap_or_else(|e| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: e.to_string(),
            },
        )
    })
}

/// Try to deserialize params, returning an error frame on failure.
pub(super) fn parse_params<T: serde::de::DeserializeOwned>(
    request_id: &str,
    params: serde_json::Value,
    method_name: &str,
) -> Result<T, Frame> {
    serde_json::from_value(params).map_err(|e| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_params".into(),
                message: format!("Invalid {method_name} params: {e}"),
            },
        )
    })
}

/// Resolve the user_id for a peer, falling back to peer_id.
pub(super) async fn resolve_user_id(state: &SharedState, peer_id: &str) -> String {
    let user_id = {
        let state_read = state.read().await;
        state_read.user_id_for_peer(peer_id)
    };
    user_id.unwrap_or_else(|| peer_id.to_string())
}

/// Run a blocking git operation, handling the git-not-configured and panic cases.
pub(super) async fn git_blocking<F, R>(
    request_id: &str,
    state: &SharedState,
    f: F,
) -> Result<R, Frame>
where
    F: FnOnce(std::sync::Arc<crate::memory::git::GitStorage>) -> anyhow::Result<R> + Send + 'static,
    R: Send + 'static,
{
    let git = state.read().await.git_storage.clone();
    let git = git.ok_or_else(|| internal_error(request_id, "Git storage not configured"))?;
    tokio::task::spawn_blocking(move || f(git))
        .await
        .map_err(|e| internal_error(request_id, format!("Task panicked: {e}")))?
        .map_err(|e| internal_error(request_id, format!("{e}")))
}

/// Build an internal_error response frame.
pub(super) fn internal_error(request_id: &str, message: impl Into<String>) -> Frame {
    Frame::error_response(
        request_id,
        ErrorPayload {
            code: "internal_error".into(),
            message: message.into(),
        },
    )
}

/// Error identity for `forward_to_node` — the error code (e.g. `"tool_unavailable"`)
/// and a human-readable label (e.g. `"Tool execution"`) used in timeout messages.
pub(super) struct ForwardErrorCodes {
    pub error_code: &'static str,
    pub label: &'static str,
}

/// Forward a request to a node peer, await the response with a timeout, and relay it back.
///
/// Accepts raw `serde_json::Value` params to avoid unnecessary deserialization and
/// re-serialization of potentially large payloads (e.g. base64 image data).
pub(super) async fn forward_to_node(
    request_id: &str,
    method: Method,
    params: serde_json::Value,
    node_sender: mpsc::Sender<Frame>,
    state: &SharedState,
    timeout: Duration,
    errors: ForwardErrorCodes,
) -> Frame {
    let forwarded_id = Frame::new_id();
    let forwarded_frame = Frame::Request {
        id: forwarded_id.clone(),
        method,
        params,
    };

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    {
        let mut sw = state.write().await;
        sw.pending_requests
            .insert(forwarded_id.clone(), response_tx);
    }

    if node_sender.send(forwarded_frame).await.is_err() {
        let mut sw = state.write().await;
        sw.pending_requests.remove(&forwarded_id);
        return Frame::error_response(
            request_id,
            ErrorPayload {
                code: errors.error_code.into(),
                message: "Failed to send request to node".into(),
            },
        );
    }

    match tokio::time::timeout(timeout, response_rx).await {
        Ok(Ok(Frame::Response {
            ok, payload, error, ..
        })) => Frame::Response {
            id: request_id.to_string(),
            ok,
            payload: if ok { payload } else { None },
            error: if ok { None } else { error },
        },
        Ok(Ok(_)) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: "Unexpected frame type from node".into(),
            },
        ),
        Ok(Err(_)) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: errors.error_code.into(),
                message: "Node disconnected during execution".into(),
            },
        ),
        Err(_) => {
            let mut sw = state.write().await;
            sw.pending_requests.remove(&forwarded_id);
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "timeout".into(),
                    message: format!("{} timed out ({}s)", errors.label, timeout.as_secs()),
                },
            )
        }
    }
}
