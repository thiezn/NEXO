//! WebSocket handler for audio generation requests.

use crate::server::state::{GatewayState, SharedState};
use nexo_core::ModelCapability;
use nexo_ws_schema::{ConnectionRole, ErrorPayload, Frame, Method};
use tokio::sync::mpsc;

use super::base::{AUDIO_GENERATION_TIMEOUT, ForwardErrorCodes, forward_to_node};

/// Forward audio.generate to a speech-generation-capable node without
/// deserializing the payload — the node validates params.
pub(super) async fn handle(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let node_sender = {
        let state_read = state.read().await;
        match find_audio_generate_sender(&state_read) {
            Some((_, sender)) => sender,
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "no_audio_generation_node".into(),
                        message: "No speech-generation-capable node is connected".into(),
                    },
                );
            }
        }
    };

    forward_to_node(
        request_id,
        Method::AudioGenerate,
        params,
        node_sender,
        state,
        AUDIO_GENERATION_TIMEOUT,
        ForwardErrorCodes {
            error_code: "audio_generation_unavailable",
            label: "Audio generation",
        },
    )
    .await
}

fn find_audio_generate_sender(state: &GatewayState) -> Option<(String, mpsc::Sender<Frame>)> {
    state.peers.iter().find_map(|(peer_id, peer)| {
        if peer.role != ConnectionRole::Node {
            return None;
        }

        let supports_audio_generation = state.loaded_models.get(peer_id).is_some_and(|models| {
            models.iter().any(|model| {
                model
                    .capabilities
                    .iter()
                    .any(|capability| matches!(capability, ModelCapability::SpeechGeneration))
            })
        });

        if !supports_audio_generation {
            return None;
        }

        state
            .peer_senders
            .get(peer_id)
            .cloned()
            .map(|sender| (peer_id.clone(), sender))
    })
}
