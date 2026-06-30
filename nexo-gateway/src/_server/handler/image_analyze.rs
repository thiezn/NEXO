//! WebSocket handler for image analysis requests.

use crate::server::state::{GatewayState, SharedState};
use nexo_core::ModelCapability;
use nexo_ws_schema::{ConnectionRole, ErrorPayload, Frame, Method};
use tokio::sync::mpsc;

use super::base::{ForwardErrorCodes, IMAGE_ANALYSIS_TIMEOUT, forward_to_node};

/// Forward image.analyze to a vision-capable node without deserializing the
/// (potentially multi-MB) base64 payload — the node validates the params.
pub(super) async fn handle(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let node_sender = {
        let state_read = state.read().await;
        match find_image_analyze_sender(&state_read) {
            Some((_, sender)) => sender,
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "no_vision_node".into(),
                        message: "No vision-capable node is connected".into(),
                    },
                );
            }
        }
    };

    forward_to_node(
        request_id,
        Method::ImageAnalyze,
        params,
        node_sender,
        state,
        IMAGE_ANALYSIS_TIMEOUT,
        ForwardErrorCodes {
            error_code: "vision_unavailable",
            label: "Image analysis",
        },
    )
    .await
}

fn find_image_analyze_sender(state: &GatewayState) -> Option<(String, mpsc::Sender<Frame>)> {
    state.peers.iter().find_map(|(peer_id, peer)| {
        if peer.role != ConnectionRole::Node {
            return None;
        }

        let supports_image_analysis = state.loaded_models.get(peer_id).is_some_and(|models| {
            models.iter().any(|model| {
                model
                    .capabilities()
                    .iter()
                    .any(|capability| matches!(capability, ModelCapability::ImageInput))
            })
        });

        if !supports_image_analysis {
            return None;
        }

        state
            .peer_senders
            .get(peer_id)
            .cloned()
            .map(|sender| (peer_id.clone(), sender))
    })
}
