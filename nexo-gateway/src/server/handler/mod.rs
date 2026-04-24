mod agent;
mod base;
mod cron;
mod image_analyze;
mod prefill;
mod send;
mod status;
mod tools;

#[cfg(test)]
mod tests;

use crate::agent::AgentHandle;
use crate::server::state::{PeerInfo, SharedState};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::{
    ConnectParams, ErrorPayload, EventKind, Frame, HelloOk, Method, PROTOCOL_VERSION,
    PresencePayload, Role, WsError,
};
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
    agent_handle: AgentHandle,
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

    // Step 2: Send presence event
    // Note: DrainQueue is triggered when we receive ModelStatus (after a model is loaded),
    // not on initial connect, since we don't know the node's loaded models yet.
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
                        if let Some(response) = handle_incoming_message(&text, &peer_id, &state, &db, &agent_handle).await {
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

    // Step 4: Cleanup — broadcast offline presence and remove peer in one lock
    {
        let mut sw = state.write().await;
        if let Some(peer) = sw.peers.get(&peer_id) {
            broadcast_presence(peer, "offline", &sw.event_tx);
        }
        sw.remove_peer(&peer_id);
    }
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
    let models = params.models;
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

    tracing::info!(
        "Peer connected: id={}, client={}, role={:?}, capabilities={:?}, commands={:?}, available_models={:?}",
        peer.id,
        peer.client_id,
        peer.role,
        peer.capabilities,
        peer.commands,
        models,
    );

    // Create per-peer directed channel
    let (directed_tx, directed_rx) = mpsc::channel(32);
    {
        let mut sw = state.write().await;
        sw.add_peer(peer, directed_tx);
        sw.set_available_models(&peer_id, models);
    }

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
    db: &SqlitePool,
    agent_handle: &AgentHandle,
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
            Some(dispatch_method(&id, &method, params, peer_id, state, db, agent_handle).await)
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

async fn dispatch_method(
    request_id: &str,
    method: &Method,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    agent_handle: &AgentHandle,
) -> Frame {
    use base::ok_or_internal_error;

    match method {
        Method::Health => status::handle_health(request_id, state).await,
        Method::Status => status::handle_status(request_id, state).await,
        Method::ToolsCatalog => status::handle_tools_catalog(request_id, state).await,
        Method::ModelStatus => {
            status::handle_model_status(request_id, params, peer_id, state, agent_handle).await
        }

        Method::ToolsRegister => tools::handle_register(request_id, params, peer_id, state).await,
        Method::ToolsExecute => tools::handle_execute(request_id, params, peer_id, state).await,

        Method::Agent => {
            agent::handle_agent(request_id, params, peer_id, state, db, agent_handle).await
        }
        Method::SessionCreate => {
            agent::handle_session_create(request_id, params, peer_id, state, db).await
        }
        Method::SessionList => agent::handle_session_list(request_id, peer_id, state, db).await,
        Method::SessionGet => agent::handle_session_get(request_id, params, db).await,
        Method::SessionClear => agent::handle_session_clear(request_id, params, db).await,

        Method::CronCreate => cron::handle_create(request_id, params, db).await,
        Method::CronList => cron::handle_list(request_id, db).await,
        Method::CronDelete => cron::handle_delete(request_id, params, db).await,

        Method::PrefillFetch => prefill::handle_fetch_deprecated(request_id),
        Method::PrefillMarkdownCreate => {
            prefill::handle_markdown_create(request_id, params, state).await
        }
        Method::PrefillMarkdownList => prefill::handle_markdown_list(request_id, state).await,
        Method::PrefillMarkdownDelete => {
            prefill::handle_markdown_delete(request_id, params, state).await
        }
        Method::PrefillCollectionCreate => {
            prefill::handle_collection_create(request_id, params, state).await
        }
        Method::PrefillCollectionList => prefill::handle_collection_list(request_id, state).await,
        Method::PrefillCollectionDelete => {
            prefill::handle_collection_delete(request_id, params, state).await
        }

        Method::ImageAnalyze => image_analyze::handle(request_id, params, state).await,

        Method::SystemPresence => {
            ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
        }
        Method::Send => send::handle_send(request_id, params, peer_id, state).await,

        Method::ModelLoad | Method::ModelUnload => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "This method is only sent by the gateway to nodes".into(),
            },
        ),
        Method::Connect => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "Connect can only be the first frame".into(),
            },
        ),
    }
}
