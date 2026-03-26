use crate::server::state::{PeerInfo, SharedState};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::{
    ConnectParams, ErrorPayload, EventKind, Frame, HealthResponse, HelloOk, Method,
    PROTOCOL_VERSION, PresencePayload, Role, StatusResponse, ToolsCatalogResponse,
    ToolsExecuteParams, ToolsRegisterParams, ToolsRegisterResponse, WsError,
};
use serde::Serialize;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};
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
    let (peer_id, _connect_request_id, mut directed_rx) =
        match wait_for_connect(&mut ws, &state, &db).await {
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

    // Step 3: Message loop (three-way select: WS messages, broadcast events, directed frames)
    loop {
        tokio::select! {
            msg = ws.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Some(response) = handle_incoming_message(&text, &peer_id, &state, &db).await {
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
            directed = directed_rx.recv() => {
                match directed {
                    Some(frame) => {
                        let json = match serde_json::to_string(&frame) {
                            Ok(j) => j,
                            Err(e) => {
                                tracing::error!("Failed to serialize directed frame: {e}");
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
/// Returns (peer_id, request_id, directed_rx).
async fn wait_for_connect<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    ws: &mut WebSocketStream<S>,
    state: &SharedState,
    db: &SqlitePool,
) -> Result<(String, String, mpsc::Receiver<Frame>), String> {
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

    // Create per-peer directed channel
    let (directed_tx, directed_rx) = mpsc::channel(32);
    state.write().await.add_peer(peer, directed_tx);

    // Send hello-ok response
    let hello = HelloOk::default();
    let response = Frame::ok_response(&request_id, &hello)
        .map_err(|e| format!("Failed to build hello-ok: {e}"))?;
    let json = serde_json::to_string(&response).map_err(|e| format!("JSON error: {e}"))?;
    ws.send(Message::Text(json.into()))
        .await
        .map_err(|e| format!("Send error: {e}"))?;

    Ok((peer_id, request_id, directed_rx))
}

/// Parse and dispatch an incoming message from a connected peer.
/// Returns `None` when the frame was a response routed to a pending request.
async fn handle_incoming_message(
    text: &str,
    peer_id: &str,
    state: &SharedState,
    _db: &SqlitePool,
) -> Option<Frame> {
    let frame: Frame = match serde_json::from_str(text) {
        Ok(f) => f,
        Err(e) => {
            return Some(Frame::error_response(
                "",
                ErrorPayload {
                    code: "parse_error".into(),
                    message: format!("Invalid JSON: {e}"),
                },
            ));
        }
    };

    match frame {
        Frame::Request { id, method, params } => {
            Some(dispatch_method(&id, &method, params, peer_id, state).await)
        }
        Frame::Response { ref id, .. } => {
            // Check if this is a response to a pending forwarded request
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
    params: serde_json::Value,
    peer_id: &str,
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
            let state = state.read().await;
            ok_or_internal_error(
                request_id,
                ToolsCatalogResponse {
                    tools: state.all_tool_entries(),
                },
            )
        }
        Method::ToolsRegister => {
            // Validate caller is a node
            {
                let state_read = state.read().await;
                match state_read.peers.get(peer_id) {
                    Some(peer) if peer.role == Role::Node => {}
                    Some(_) => {
                        return Frame::error_response(
                            request_id,
                            ErrorPayload {
                                code: "forbidden".into(),
                                message: "Only nodes can register tools".into(),
                            },
                        );
                    }
                    None => {
                        return Frame::error_response(
                            request_id,
                            ErrorPayload {
                                code: "unknown_peer".into(),
                                message: "Peer not found in state".into(),
                            },
                        );
                    }
                }
            }

            let register_params: ToolsRegisterParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return Frame::error_response(
                        request_id,
                        ErrorPayload {
                            code: "invalid_params".into(),
                            message: format!("Invalid tools.register params: {e}"),
                        },
                    );
                }
            };

            let tool_count = register_params.tools.len();
            let registered = {
                let mut state_write = state.write().await;
                state_write.register_tools(peer_id, register_params.tools)
            };

            tracing::info!(
                "Node {peer_id} registered {registered}/{tool_count} tool(s)"
            );

            ok_or_internal_error(request_id, ToolsRegisterResponse { registered })
        }
        Method::ToolsExecute => {
            let exec_params: ToolsExecuteParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return Frame::error_response(
                        request_id,
                        ErrorPayload {
                            code: "invalid_params".into(),
                            message: format!("Invalid tools.execute params: {e}"),
                        },
                    );
                }
            };

            tracing::info!(
                "Routing tools.execute for '{}' (requested by peer {peer_id})",
                exec_params.tool
            );

            // Look up the tool and get the node's sender
            let (node_sender, forwarded_id) = {
                let state_read = state.read().await;
                let tool = match state_read.find_tool(&exec_params.tool) {
                    Some(t) => t,
                    None => {
                        return Frame::error_response(
                            request_id,
                            ErrorPayload {
                                code: "tool_not_found".into(),
                                message: format!(
                                    "Tool '{}' is not registered",
                                    exec_params.tool
                                ),
                            },
                        );
                    }
                };
                let sender = match state_read.peer_senders.get(&tool.peer_id) {
                    Some(s) => s.clone(),
                    None => {
                        return Frame::error_response(
                            request_id,
                            ErrorPayload {
                                code: "tool_unavailable".into(),
                                message: format!(
                                    "Node hosting tool '{}' is not connected",
                                    exec_params.tool
                                ),
                            },
                        );
                    }
                };
                let fwd_id = Frame::new_id();
                (sender, fwd_id)
            };

            // Build forwarded request frame for the node
            let forwarded_frame = match Frame::request(Method::ToolsExecute, &exec_params) {
                Ok(mut f) => {
                    // Override the id so we can match the response
                    if let Frame::Request { ref mut id, .. } = f {
                        *id = forwarded_id.clone();
                    }
                    f
                }
                Err(e) => {
                    return Frame::error_response(
                        request_id,
                        ErrorPayload {
                            code: "internal_error".into(),
                            message: format!("Failed to build forwarded request: {e}"),
                        },
                    );
                }
            };

            // Create oneshot for awaiting the node's response
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            {
                let mut state_write = state.write().await;
                state_write
                    .pending_requests
                    .insert(forwarded_id.clone(), response_tx);
            }

            // Send to node via directed channel
            if node_sender.send(forwarded_frame).await.is_err() {
                let mut state_write = state.write().await;
                state_write.pending_requests.remove(&forwarded_id);
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "tool_unavailable".into(),
                        message: "Failed to send request to node".into(),
                    },
                );
            }

            tracing::debug!(
                "Forwarded tools.execute to node (forwarded_id={forwarded_id})"
            );

            // Await response with timeout
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                response_rx,
            )
            .await
            {
                Ok(Ok(Frame::Response {
                    ok, payload, error, ..
                })) => {
                    // Relay the node's response back to the user with the original request id
                    if ok {
                        Frame::Response {
                            id: request_id.to_string(),
                            ok: true,
                            payload,
                            error: None,
                        }
                    } else {
                        Frame::Response {
                            id: request_id.to_string(),
                            ok: false,
                            payload: None,
                            error,
                        }
                    }
                }
                Ok(Ok(_)) => Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "internal_error".into(),
                        message: "Unexpected frame type from node".into(),
                    },
                ),
                Ok(Err(_)) => {
                    // Oneshot sender dropped (node disconnected)
                    Frame::error_response(
                        request_id,
                        ErrorPayload {
                            code: "tool_unavailable".into(),
                            message: "Node disconnected during tool execution".into(),
                        },
                    )
                }
                Err(_) => {
                    // Timeout
                    let mut state_write = state.write().await;
                    state_write.pending_requests.remove(&forwarded_id);
                    Frame::error_response(
                        request_id,
                        ErrorPayload {
                            code: "timeout".into(),
                            message: "Tool execution timed out (30s)".into(),
                        },
                    )
                }
            }
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
    use crate::server::state::{GatewayState, dummy_sender};
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
            serde_json::json!({}),
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
            s.add_peer(
                PeerInfo {
                    id: "p1".into(),
                    client_id: "cli".into(),
                    role: Role::User,
                    scopes: vec![],
                    capabilities: vec![],
                    commands: vec![],
                    device_id: None,
                    connected_at: chrono::Utc::now(),
                },
                dummy_sender(),
            );
        }
        let resp = dispatch_method(
            "req-1",
            &Method::Status,
            serde_json::json!({}),
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
            serde_json::json!({}),
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

    #[tokio::test]
    async fn dispatch_tools_register_from_node() {
        let state = make_state();
        {
            let mut s = state.write().await;
            s.add_peer(
                PeerInfo {
                    id: "n1".into(),
                    client_id: "node".into(),
                    role: Role::Node,
                    scopes: vec![],
                    capabilities: vec!["echo".into()],
                    commands: vec!["echo.run".into()],
                    device_id: None,
                    connected_at: chrono::Utc::now(),
                },
                dummy_sender(),
            );
        }

        let params = serde_json::json!({
            "tools": [{
                "name": "echo.run",
                "description": "Echo input",
                "parameters": {"type": "object"}
            }]
        });

        let resp = dispatch_method("req-1", &Method::ToolsRegister, params, "n1", &state).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            assert_eq!(payload.unwrap()["registered"], 1);
        } else {
            panic!("Expected response");
        }

        // Verify tool is in catalog
        let catalog_resp = dispatch_method(
            "req-2",
            &Method::ToolsCatalog,
            serde_json::json!({}),
            "n1",
            &state,
        )
        .await;
        if let Frame::Response { ok, payload, .. } = catalog_resp {
            assert!(ok);
            let tools = &payload.unwrap()["tools"];
            assert_eq!(tools.as_array().unwrap().len(), 1);
            assert_eq!(tools[0]["name"], "echo.run");
        } else {
            panic!("Expected response");
        }
    }

    #[tokio::test]
    async fn dispatch_tools_register_from_user_rejected() {
        let state = make_state();
        {
            let mut s = state.write().await;
            s.add_peer(
                PeerInfo {
                    id: "u1".into(),
                    client_id: "cli".into(),
                    role: Role::User,
                    scopes: vec![],
                    capabilities: vec![],
                    commands: vec![],
                    device_id: None,
                    connected_at: chrono::Utc::now(),
                },
                dummy_sender(),
            );
        }

        let params = serde_json::json!({"tools": []});
        let resp = dispatch_method("req-1", &Method::ToolsRegister, params, "u1", &state).await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "forbidden");
        } else {
            panic!("Expected error response");
        }
    }

    #[tokio::test]
    async fn dispatch_tools_execute_tool_not_found() {
        let state = make_state();
        let params = serde_json::json!({
            "tool": "nonexistent",
            "args": {},
            "idempotencyKey": "k1"
        });
        let resp =
            dispatch_method("req-1", &Method::ToolsExecute, params, "u1", &state).await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "tool_not_found");
        } else {
            panic!("Expected error response");
        }
    }
}
