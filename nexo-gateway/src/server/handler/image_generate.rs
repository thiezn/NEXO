//! WebSocket handler for image generation requests.

use crate::server::state::{GatewayState, SharedState};
use nexo_core::ModelCapability;
use nexo_ws_schema::{ConnectionRole, ErrorPayload, Frame, Method};
use tokio::sync::mpsc;

use super::base::{ForwardErrorCodes, IMAGE_GENERATION_TIMEOUT, forward_to_node};

/// Forward image.generate to an image-generation-capable node without
/// deserializing the payload — the node validates params.
pub(super) async fn handle(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let node_sender = {
        let state_read = state.read().await;
        match find_image_generate_sender(&state_read) {
            Some((_, sender)) => sender,
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "no_image_generation_node".into(),
                        message: "No image-generation-capable node is connected".into(),
                    },
                );
            }
        }
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

fn find_image_generate_sender(state: &GatewayState) -> Option<(String, mpsc::Sender<Frame>)> {
    state.peers.iter().find_map(|(peer_id, peer)| {
        if peer.role != ConnectionRole::Node {
            return None;
        }

        let supports_image_generation = state.loaded_models.get(peer_id).is_some_and(|models| {
            models.iter().any(|model| {
                model
                    .capabilities
                    .iter()
                    .any(|capability| matches!(capability, ModelCapability::ImageGeneration))
            })
        });

        if !supports_image_generation {
            return None;
        }

        state
            .peer_senders
            .get(peer_id)
            .cloned()
            .map(|sender| (peer_id.clone(), sender))
    })
}
