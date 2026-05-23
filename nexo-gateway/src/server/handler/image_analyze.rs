//! WebSocket handler for image analysis requests.

use crate::server::state::SharedState;
use nexo_ws_schema::{ErrorPayload, Frame, Method};

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
        match state_read.find_image_analyze_peer() {
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
