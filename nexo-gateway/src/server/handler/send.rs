use crate::server::state::SharedState;
use nexo_ws_schema::{
    ErrorPayload, EventKind, Frame, MessagePayload, Role, SendParams, SendResponse,
};

use super::base::{ok_or_internal_error, parse_params};

pub(super) async fn handle_send(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
) -> Frame {
    let send_params: SendParams = match parse_params(request_id, params, "send") {
        Ok(p) => p,
        Err(f) => return f,
    };

    if send_params.target.trim().is_empty() {
        return Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_params".into(),
                message: "Send target cannot be empty".into(),
            },
        );
    }

    let (sender_client_id, recipients) = {
        let state_read = state.read().await;
        let peer = match state_read.peers.get(peer_id) {
            Some(peer) => peer,
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "unknown_peer".into(),
                        message: "Peer not found in state".into(),
                    },
                );
            }
        };

        if peer.role != Role::User {
            return Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "forbidden".into(),
                    message: "Only user clients can send client messages".into(),
                },
            );
        }

        (
            peer.client_id.clone(),
            state_read.find_user_peers_by_client_id(&send_params.target, peer_id),
        )
    };

    let event = match Frame::event(
        EventKind::Message,
        MessagePayload {
            message_id: Frame::new_id(),
            from: sender_client_id.clone(),
            target: send_params.target.clone(),
            payload: send_params.payload,
        },
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "internal_error".into(),
                    message: error.to_string(),
                },
            );
        }
    };

    let mut delivered = 0usize;
    for (target_peer_id, sender) in recipients {
        if sender.send(event.clone()).await.is_ok() {
            delivered += 1;
            tracing::debug!(
                "Delivered message from {} to peer {} for target {}",
                sender_client_id,
                target_peer_id,
                send_params.target,
            );
        } else {
            tracing::warn!(
                "Failed to deliver message from {} to peer {} for target {}",
                sender_client_id,
                target_peer_id,
                send_params.target,
            );
        }
    }

    tracing::info!(
        "Processed send from {} to {} with {} delivery(ies)",
        sender_client_id,
        send_params.target,
        delivered,
    );

    ok_or_internal_error(
        request_id,
        SendResponse {
            delivered: delivered > 0,
        },
    )
}
