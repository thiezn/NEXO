use crate::server::state::{PeerInfo, SharedState};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::{
    ConnectParams, ErrorPayload, EventKind, Frame, HealthResponse, HelloOk, Method,
    PROTOCOL_VERSION, PresencePayload, Role, StatusResponse, ToolsCatalogResponse, WsError,
};
use serde::Serialize;
use sqlx::SqlitePool;
use tokio::sync::broadcast;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

/// Broadcast a presence event for a peer.
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

/// Handle a single WebSocket connection from accept to close.
pub async fn handle_connection<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    mut ws: WebSocketStream<S>,
    state: SharedState,
    db: SqlitePool,
    mut event_rx: broadcast::Receiver<Frame>,
) {
    // Step 1: Wait for the connect frame (first frame must be connect)
    let (peer_id, _connect_request_id) = match wait_for_connect(&mut ws, &state, &db).await {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("Connection rejected: {e}");
            let _ = ws.close(None).await;
            return;
        }
    };

    // Step 2: Send presence event for the new peer
    {
        let state_read = state.read().await;
        if let Some(peer) = state_read.peers.get(&peer_id) {
            broadcast_presence(peer, "online", &state_read.event_tx);
        }
    }

    // Step 3: Message loop
    loop {
        tokio::select! {
            msg = ws.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let response = handle_incoming_message(&text, &peer_id, &state, &db).await;
                        let json = match serde_json::to_string(&response) {
                            Ok(j) => j,
                            Err(e) => {
                                tracing::error!("Failed to serialize response: {e}");
                                continue;
                            }
                        };
                        if ws.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => continue, // Ignore ping/pong/binary
                    Some(Err(e)) => {
                        tracing::debug!("WS error from peer {peer_id}: {e}");
                        break;
                    }
                }
            }
            event = event_rx.recv() => {
                match event {
                    Ok(frame) => {
                        let json = match serde_json::to_string(&frame) {
                            Ok(j) => j,
                            Err(_) => continue,
                        };
                        if ws.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("Peer {peer_id} lagged by {n} events");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    // Step 4: Cleanup — gather presence info under read lock, then write lock only for removal
    let offline_peer = {
        let state_read = state.read().await;
        state_read
            .peers
            .get(&peer_id)
            .map(|peer| (peer.client_id.clone(), peer.role))
    };
    if let Some((client_id, role)) = offline_peer {
        let presence = PresencePayload {
            client_id,
            role,
            status: "offline".into(),
        };
        if let Ok(frame) = Frame::event(EventKind::Presence, &presence) {
            let state_read = state.read().await;
            let _ = state_read.event_tx.send(frame);
        }
    }
    state.write().await.remove_peer(&peer_id);
}

/// Wait for the first connect frame, validate it, register the peer, and send hello-ok.
async fn wait_for_connect<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    ws: &mut WebSocketStream<S>,
    state: &SharedState,
    db: &SqlitePool,
) -> Result<(String, String), String> {
    let msg = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
        .await
        .map_err(|_| "Timeout waiting for connect frame".to_string())?
        .ok_or_else(|| "Connection closed before connect".to_string())?
        .map_err(|e| format!("WS error: {e}"))?;

    let text = match msg {
        Message::Text(t) => t,
        _ => return Err("First frame must be a text frame".into()),
    };

    let frame: Frame =
        serde_json::from_str(&text).map_err(|e| format!("Invalid JSON frame: {e}"))?;

    let (request_id, params_value) = match frame {
        Frame::Request {
            id,
            method: Method::Connect,
            params,
        } => (id, params),
        _ => return Err("First frame must be a connect request".into()),
    };

    let params: ConnectParams =
        serde_json::from_value(params_value).map_err(|e| format!("Invalid connect params: {e}"))?;

    // Validate protocol version
    if params.min_protocol > PROTOCOL_VERSION || params.max_protocol < PROTOCOL_VERSION {
        let err = WsError::ProtocolMismatch {
            min: params.min_protocol,
            max: params.max_protocol,
            server: PROTOCOL_VERSION,
        };
        let error_frame = Frame::error_response(&request_id, ErrorPayload::from(&err));
        let json = serde_json::to_string(&error_frame).unwrap_or_default();
        let _ = ws.send(Message::Text(json.into())).await;
        return Err(err.to_string());
    }

    // Register device in persistent store
    if let Some(ref device) = params.device {
        if let Err(e) = crate::memory::persistent::upsert_device(db, &device.id, params.role).await
        {
            tracing::warn!("Failed to persist device: {e}");
        }
        if params.role == Role::User
            && let Err(e) =
                crate::memory::persistent::upsert_user(db, &params.client.id, &device.id).await
        {
            tracing::warn!("Failed to persist user: {e}");
        }
    }

    let peer_id = Frame::new_id();
    let peer = PeerInfo {
        id: peer_id.clone(),
        client_id: params.client.id.clone(),
        role: params.role,
        scopes: params.scopes,
        capabilities: params.capabilities,
        commands: params.commands,
        device_id: params.device.map(|d| d.id),
        connected_at: chrono::Utc::now(),
    };

    state.write().await.add_peer(peer);

    // Send hello-ok response
    let hello = HelloOk::default();
    let response = Frame::ok_response(&request_id, &hello)
        .map_err(|e| format!("Failed to build hello-ok: {e}"))?;
    let json = serde_json::to_string(&response).map_err(|e| format!("JSON error: {e}"))?;
    ws.send(Message::Text(json.into()))
        .await
        .map_err(|e| format!("Send error: {e}"))?;

    Ok((peer_id, request_id))
}

/// Parse and dispatch an incoming message from a connected peer.
async fn handle_incoming_message(
    text: &str,
    peer_id: &str,
    state: &SharedState,
    _db: &SqlitePool,
) -> Frame {
    let frame: Frame = match serde_json::from_str(text) {
        Ok(f) => f,
        Err(e) => {
            return Frame::error_response(
                "",
                ErrorPayload {
                    code: "parse_error".into(),
                    message: format!("Invalid JSON: {e}"),
                },
            );
        }
    };

    match frame {
        Frame::Request { id, method, params } => {
            dispatch_method(&id, &method, &params, peer_id, state).await
        }
        _ => Frame::error_response(
            "",
            ErrorPayload {
                code: "invalid_frame".into(),
                message: "Expected request frame".into(),
            },
        ),
    }
}

/// Build an ok response, falling back to an internal_error response on serialization failure.
fn ok_or_internal_error(request_id: &str, payload: impl Serialize) -> Frame {
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

async fn dispatch_method(
    request_id: &str,
    method: &Method,
    _params: &serde_json::Value,
    _peer_id: &str,
    state: &SharedState,
) -> Frame {
    match method {
        Method::Health => {
            let state = state.read().await;
            ok_or_internal_error(
                request_id,
                HealthResponse {
                    status: "ok".into(),
                    uptime_secs: state.uptime_secs(),
                },
            )
        }
        Method::Status => {
            let state = state.read().await;
            ok_or_internal_error(
                request_id,
                StatusResponse {
                    connected_users: state.connected_users(),
                    connected_nodes: state.connected_nodes(),
                    capabilities: state.all_capabilities(),
                },
            )
        }
        Method::ToolsCatalog => {
            ok_or_internal_error(request_id, ToolsCatalogResponse { tools: vec![] })
        }
        Method::Agent => {
            let run_id = Frame::new_id();
            ok_or_internal_error(
                request_id,
                nexo_ws_schema::AgentResponse {
                    run_id,
                    status: "accepted".into(),
                    summary: None,
                },
            )
        }
        Method::SystemPresence => {
            ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
        }
        Method::Send => {
            ok_or_internal_error(request_id, nexo_ws_schema::SendResponse { delivered: true })
        }
        Method::Connect => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "Connect can only be the first frame".into(),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    use crate::server::state::GatewayState;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_state() -> SharedState {
        Arc::new(RwLock::new(GatewayState::new()))
    }

    #[tokio::test]
    async fn dispatch_health_returns_ok() {
        let state = make_state();
        let resp = dispatch_method(
            "req-1",
            &Method::Health,
            &serde_json::json!({}),
            "p1",
            &state,
        )
        .await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert_eq!(p["status"], "ok");
        } else {
            panic!("Expected response");
        }
    }

    #[tokio::test]
    async fn dispatch_status_returns_counts() {
        let state = make_state();
        {
            let mut s = state.write().await;
            s.add_peer(PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: Role::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            });
        }
        let resp = dispatch_method(
            "req-1",
            &Method::Status,
            &serde_json::json!({}),
            "p1",
            &state,
        )
        .await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert_eq!(p["connectedUsers"], 1);
            assert_eq!(p["connectedNodes"], 0);
        } else {
            panic!("Expected response");
        }
    }

    #[tokio::test]
    async fn dispatch_connect_after_handshake_rejected() {
        let state = make_state();
        let resp = dispatch_method(
            "req-1",
            &Method::Connect,
            &serde_json::json!({}),
            "p1",
            &state,
        )
        .await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "invalid_method");
        } else {
            panic!("Expected error response");
        }
    }
}
