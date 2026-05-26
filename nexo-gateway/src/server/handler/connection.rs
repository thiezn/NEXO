//! Connection lifecycle handling for a single WebSocket peer.

use super::dispatch_method;
use crate::agent::RunHandle;
use crate::server::state::{PeerInfo, SharedState};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::{
    ConnectParams, ConnectionRole, ErrorPayload, EventKind, Frame, HelloOk, Method,
    PROTOCOL_VERSION, PresencePayload, WsError,
};
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

/// Broadcast a presence event for a connected peer.
fn broadcast_presence(peer: &PeerInfo, status: &str, event_tx: &broadcast::Sender<Frame>) {
    let presence = PresencePayload {
        client_id: peer.client_id.clone(),
        role: peer.role,
        status: status.into(),
    };
    if let Ok(frame) = Frame::event(EventKind::Presence, &presence) {
        let _ = event_tx.send(frame);
    }
}

/// Handle a single WebSocket connection from handshake completion to disconnect.
pub async fn handle_connection<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    mut ws: WebSocketStream<S>,
    state: SharedState,
    db: SqlitePool,
    mut event_rx: broadcast::Receiver<Frame>,
    run_handle: RunHandle,
) {
    let (peer_id, _connect_request_id, mut directed_rx) =
        match wait_for_connect(&mut ws, &state, &db).await {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!("Connection rejected: {error}");
                let _ = ws.close(None).await;
                return;
            }
        };

    {
        let state_read = state.read().await;
        if let Some(peer) = state_read.peers.get(&peer_id) {
            broadcast_presence(peer, "online", &state_read.event_tx);
        }
    }

    loop {
        tokio::select! {
            msg = ws.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Some(response) = handle_incoming_message(&text, &peer_id, &state, &db, &run_handle).await {
                            let json = match serde_json::to_string(&response) {
                                Ok(json) => json,
                                Err(error) => {
                                    tracing::error!("Failed to serialize response: {error}");
                                    continue;
                                }
                            };
                            if ws.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => continue,
                    Some(Err(error)) => {
                        tracing::debug!("WS error from peer {peer_id}: {error}");
                        break;
                    }
                }
            }
            event = event_rx.recv() => {
                match event {
                    Ok(frame) => {
                        let json = match serde_json::to_string(&frame) {
                            Ok(json) => json,
                            Err(_) => continue,
                        };
                        if ws.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(count)) => {
                        tracing::warn!("Peer {peer_id} lagged by {count} events");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            directed = directed_rx.recv() => {
                match directed {
                    Some(frame) => {
                        let json = match serde_json::to_string(&frame) {
                            Ok(json) => json,
                            Err(error) => {
                                tracing::error!("Failed to serialize directed frame: {error}");
                                continue;
                            }
                        };
                        tracing::debug!("Sending directed frame to peer {peer_id}");
                        if ws.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    None => {
                        tracing::debug!("Directed channel closed for peer {peer_id}");
                        break;
                    }
                }
            }
        }
    }

    {
        let mut state_write = state.write().await;
        if let Some(peer) = state_write.peers.get(&peer_id) {
            broadcast_presence(peer, "offline", &state_write.event_tx);
        }
        state_write.remove_peer(&peer_id);
    }
}

/// Wait for the initial `connect` request, register the peer, and reply with `hello`.
async fn wait_for_connect<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    ws: &mut WebSocketStream<S>,
    state: &SharedState,
    db: &SqlitePool,
) -> Result<(String, String, mpsc::Receiver<Frame>), String> {
    let msg = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
        .await
        .map_err(|_| "Timeout waiting for connect frame".to_string())?
        .ok_or_else(|| "Connection closed before connect".to_string())?
        .map_err(|error| format!("WS error: {error}"))?;

    let text = match msg {
        Message::Text(text) => text,
        _ => return Err("First frame must be a text frame".into()),
    };

    let frame: Frame =
        serde_json::from_str(&text).map_err(|error| format!("Invalid JSON frame: {error}"))?;

    let (request_id, params_value) = match frame {
        Frame::Request {
            id,
            method: Method::Connect,
            params,
        } => (id, params),
        _ => return Err("First frame must be a connect request".into()),
    };

    let params: ConnectParams = serde_json::from_value(params_value)
        .map_err(|error| format!("Invalid connect params: {error}"))?;

    if params.min_protocol > PROTOCOL_VERSION || params.max_protocol < PROTOCOL_VERSION {
        let error = WsError::ProtocolMismatch {
            min: params.min_protocol,
            max: params.max_protocol,
            server: PROTOCOL_VERSION,
        };
        let error_frame = Frame::error_response(&request_id, ErrorPayload::from(&error));
        let json = serde_json::to_string(&error_frame).unwrap_or_default();
        let _ = ws.send(Message::Text(json.into())).await;
        return Err(error.to_string());
    }

    if let Some(ref device) = params.device {
        if let Err(error) =
            crate::memory::persistent::upsert_device(db, &device.id, params.role).await
        {
            tracing::warn!("Failed to persist device: {error}");
        }
        if params.role == ConnectionRole::User
            && let Err(error) =
                crate::memory::persistent::upsert_user(db, &params.client.id, &device.id).await
        {
            tracing::warn!("Failed to persist user: {error}");
        }
    }

    let peer_id = Frame::new_id();
    let models = params.models;
    let peer = PeerInfo {
        id: peer_id.clone(),
        client_id: params.client.id.clone(),
        role: params.role,
        scopes: params.scopes,
        capabilities: params.capabilities,
        commands: params.commands,
        device_id: params.device.map(|device| device.id),
        connected_at: chrono::Utc::now(),
    };

    tracing::info!(
        "Peer connected: id={}, client={}, role={:?}, capabilities={:?}, commands={:?}, available_models={:?}",
        peer.id,
        peer.client_id,
        peer.role,
        peer.capabilities,
        peer.commands,
        models,
    );

    let (directed_tx, directed_rx) = mpsc::channel(32);
    {
        let mut state_write = state.write().await;
        state_write.add_peer(peer, directed_tx);
        state_write.set_available_models(&peer_id, models);
    }

    let hello = HelloOk::default();
    let response = Frame::ok_response(&request_id, &hello)
        .map_err(|error| format!("Failed to build hello-ok: {error}"))?;
    let json = serde_json::to_string(&response).map_err(|error| format!("JSON error: {error}"))?;
    ws.send(Message::Text(json.into()))
        .await
        .map_err(|error| format!("Send error: {error}"))?;

    Ok((peer_id, request_id, directed_rx))
}

/// Parse a single incoming frame and either dispatch it or route a pending response.
async fn handle_incoming_message(
    text: &str,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    run_handle: &RunHandle,
) -> Option<Frame> {
    let frame: Frame = match serde_json::from_str(text) {
        Ok(frame) => frame,
        Err(error) => {
            return Some(Frame::error_response(
                "",
                ErrorPayload {
                    code: "parse_error".into(),
                    message: format!("Invalid JSON: {error}"),
                },
            ));
        }
    };

    match frame {
        Frame::Request { id, method, params } => {
            Some(dispatch_method(&id, &method, params, peer_id, state, db, run_handle).await)
        }
        Frame::Response { ref id, .. } => {
            let sender = {
                let mut state_write = state.write().await;
                state_write.pending_requests.remove(id)
            };
            if let Some(tx) = sender {
                tracing::debug!("Routing response {id} to pending request");
                let _ = tx.send(frame);
                None
            } else {
                let id = id.clone();
                Some(Frame::error_response(
                    &id,
                    ErrorPayload {
                        code: "unexpected_response".into(),
                        message: "No pending request for this response".into(),
                    },
                ))
            }
        }
        _ => Some(Frame::error_response(
            "",
            ErrorPayload {
                code: "invalid_frame".into(),
                message: "Expected request frame".into(),
            },
        )),
    }
}
