//! WebSocket handler for image generation requests.

use crate::agent::r#loop::router::{RouteError, Router};
use crate::server::state::SharedState;
use nexo_core::ModelCapability;
use nexo_ws_schema::{ErrorPayload, EventKind, Frame, MessagePayload, Method};

use super::base::{ForwardErrorCodes, IMAGE_GENERATION_TIMEOUT, forward_to_node};

/// Forward image.generate to an image-generation-capable node without
/// deserializing the payload — the node validates params.
pub(super) async fn handle(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let session_id = params
        .get("sessionId")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default")
        .to_string();

    let mut is_queued = false;
    let node_sender = loop {
        let notify = {
            let state_read = state.read().await;
            state_read.model_ready_notify.clone()
        };

        match Router::route_capability(state, ModelCapability::ImageGeneration).await {
            Ok((_peer_id, sender)) => {
                if is_queued {
                    let remaining = {
                        let mut state_write = state.write().await;
                        state_write.decrement_generation_queue(&session_id)
                    };
                    tracing::info!(
                        request_id,
                        session_id,
                        queued_count = remaining,
                        "Resuming queued image.generate request"
                    );
                }
                break sender;
            }
            Err(RouteError::Error(message)) => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "image_generation_unavailable".into(),
                        message,
                    },
                );
            }
            Err(RouteError::NoCapableNode) => {}
        }

        if !is_queued {
            let queued_count = {
                let mut state_write = state.write().await;
                state_write.increment_generation_queue(&session_id)
            };

            emit_generation_queued_event(
                state,
                "image.generate",
                request_id,
                &session_id,
                queued_count,
            )
            .await;
            tracing::info!(
                request_id,
                session_id,
                queued_count,
                "Queued image.generate request waiting for image-generation-capable node"
            );
            is_queued = true;
        }

        notify.notified().await;
    };

    forward_to_node(
        request_id,
        Method::ImageGenerate,
        params,
        node_sender,
        state,
        IMAGE_GENERATION_TIMEOUT,
        ForwardErrorCodes {
            error_code: "image_generation_unavailable",
            label: "Image generation",
        },
    )
    .await
}

async fn emit_generation_queued_event(
    state: &SharedState,
    method: &str,
    request_id: &str,
    session_id: &str,
    queued_count: usize,
) {
    let event_tx = state.read().await.event_tx.clone();
    let payload = MessagePayload {
        message_id: Frame::new_id(),
        from: "gateway".to_string(),
        target: session_id.to_string(),
        payload: serde_json::json!({
            "kind": "generation.queued",
            "method": method,
            "requestId": request_id,
            "sessionId": session_id,
            "queuedCount": queued_count,
        }),
    };
    if let Ok(frame) = Frame::event(EventKind::Message, &payload) {
        let _ = event_tx.send(frame);
    }
}
