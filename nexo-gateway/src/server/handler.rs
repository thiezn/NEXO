use crate::agent::{AgentCommand, AgentHandle};
use crate::server::state::{PeerInfo, SharedState};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::{
    AgentParams, AgentStatus, ConnectParams, CronCreateParams, CronDeleteParams, ErrorPayload,
    EventKind, Frame, HealthResponse, HelloOk, Method, ModelStatusParams, PROTOCOL_VERSION,
    PrefillCollectionCreateParams, PrefillCollectionDeleteParams, PrefillFetchParams,
    PrefillMarkdownCreateParams, PrefillMarkdownDeleteParams, PresencePayload, Role,
    SessionClearParams, SessionCreateParams, SessionGetParams, StatusResponse, ToolsCatalogResponse,
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

    // Step 2: Send presence event for the new peer
    {
        let state_read = state.read().await;
        if let Some(peer) = state_read.peers.get(&peer_id) {
            broadcast_presence(peer, "online", &state_read.event_tx);
        }
    }

    // Step 2b: Drain the queue if an LLM-capable node just connected
    {
        let is_llm_node = {
            let state_read = state.read().await;
            state_read.peers.get(&peer_id).is_some_and(|p| {
                p.role == Role::Node
                    && p.capabilities.iter().any(|c| c == "llm" || c == "inference")
            })
        };
        if is_llm_node {
            if let Err(e) = agent_handle.submit(AgentCommand::DrainQueue).await {
                tracing::warn!("Failed to submit DrainQueue after LLM node connect: {e}");
            }
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

/// Try to deserialize params, returning an error frame on failure.
fn parse_params<T: serde::de::DeserializeOwned>(
    request_id: &str,
    params: serde_json::Value,
    method_name: &str,
) -> Result<T, Frame> {
    serde_json::from_value(params).map_err(|e| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_params".into(),
                message: format!("Invalid {method_name} params: {e}"),
            },
        )
    })
}

/// Resolve the user_id for a peer, falling back to peer_id.
async fn resolve_user_id(state: &SharedState, peer_id: &str) -> String {
    let user_id = {
        let state_read = state.read().await;
        state_read.user_id_for_peer(peer_id)
    };
    user_id.unwrap_or_else(|| peer_id.to_string())
}

/// Build an internal_error response frame.
fn internal_error(request_id: &str, message: impl Into<String>) -> Frame {
    Frame::error_response(
        request_id,
        ErrorPayload {
            code: "internal_error".into(),
            message: message.into(),
        },
    )
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

            let register_params: ToolsRegisterParams = match parse_params(request_id, params, "tools.register") {
                Ok(p) => p,
                Err(f) => return f,
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
            let exec_params: ToolsExecuteParams = match parse_params(request_id, params, "tools.execute") {
                Ok(p) => p,
                Err(f) => return f,
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
            let agent_params: AgentParams = match parse_params(request_id, params, "agent") {
                Ok(p) => p,
                Err(f) => return f,
            };

            let user_id = resolve_user_id(state, peer_id).await;

            // Resolve or create session
            let session_id = match agent_params.session_id {
                Some(sid) => sid,
                None => {
                    match crate::agent::session::create_session(db, &user_id, None, None::<&str>).await {
                        Ok((sid, _)) => sid,
                        Err(e) => return internal_error(request_id, format!("Failed to create session: {e}")),
                    }
                }
            };

            // Look up the session's prefill_collection_id
            let prefill_collection_id: Option<String> =
                sqlx::query_as::<_, (Option<String>,)>(
                    "SELECT prefill_collection_id FROM sessions WHERE id = ?",
                )
                .bind(&session_id)
                .fetch_optional(db)
                .await
                .ok()
                .flatten()
                .and_then(|(c,)| c);

            let run_id = Frame::new_id();
            if let Err(e) = crate::agent::session::create_run(
                db,
                &run_id,
                &session_id,
                &agent_params.idempotency_key,
                agent_params.model_id.as_deref(),
            )
            .await
            {
                return internal_error(request_id, format!("Failed to create run: {e}"));
            }

            // Submit to agent background task
            let cmd = AgentCommand::RunAgent {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                prompt: agent_params.prompt,
                context: agent_params.context,
                peer_id: peer_id.to_string(),
                model_id: agent_params.model_id,
                prefill_collection_id,
            };
            if let Err(e) = agent_handle.submit(cmd).await {
                tracing::error!("Failed to submit agent command: {e}");
            }

            ok_or_internal_error(
                request_id,
                nexo_ws_schema::AgentResponse {
                    run_id,
                    session_id,
                    status: AgentStatus::Accepted,
                    summary: None,
                },
            )
        }
        Method::SessionCreate => {
            let session_params: SessionCreateParams = match parse_params(request_id, params, "session.create") {
                Ok(p) => p,
                Err(f) => return f,
            };
            let user_id = resolve_user_id(state, peer_id).await;

            match crate::agent::session::create_session(
                db,
                &user_id,
                session_params.name.as_deref(),
                session_params.prefill_collection_id.as_deref(),
            )
            .await
            {
                Ok((session_id, prefill_collection_id)) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::SessionCreateResponse { session_id, prefill_collection_id },
                ),
                Err(e) => internal_error(request_id, format!("Failed to create session: {e}")),
            }
        }
        Method::SessionList => {
            let user_id = resolve_user_id(state, peer_id).await;

            match crate::agent::session::list_sessions(db, &user_id).await {
                Ok(sessions) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::SessionListResponse { sessions },
                ),
                Err(e) => internal_error(request_id, format!("Failed to list sessions: {e}")),
            }
        }
        Method::SessionGet => {
            let get_params: SessionGetParams = match parse_params(request_id, params, "session.get") {
                Ok(p) => p,
                Err(f) => return f,
            };

            match crate::agent::session::get_session(db, &get_params.session_id).await {
                Ok(Some(resp)) => ok_or_internal_error(request_id, resp),
                Ok(None) => Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "session_not_found".into(),
                        message: format!("Session '{}' not found", get_params.session_id),
                    },
                ),
                Err(e) => internal_error(request_id, format!("Failed to get session: {e}")),
            }
        }
        Method::SessionClear => {
            let clear_params: SessionClearParams = match parse_params(request_id, params, "session.clear") {
                Ok(p) => p,
                Err(f) => return f,
            };

            match crate::agent::session::clear_session(db, &clear_params.session_id).await {
                Ok(cleared) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::SessionClearResponse { cleared },
                ),
                Err(e) => internal_error(request_id, format!("Failed to clear session: {e}")),
            }
        }
        Method::CronCreate => {
            let cron_params: CronCreateParams = match parse_params(request_id, params, "cron.create") {
                Ok(p) => p,
                Err(f) => return f,
            };

            match crate::agent::cron::create_job(
                db, &cron_params.name, &cron_params.schedule, &cron_params.prompt,
                cron_params.session_id.as_deref(),
            ).await {
                Ok(job_id) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::CronCreateResponse { job_id },
                ),
                Err(e) => internal_error(request_id, format!("Failed to create cron job: {e}")),
            }
        }
        Method::CronList => match crate::agent::cron::list_jobs(db).await {
            Ok(jobs) => {
                ok_or_internal_error(request_id, nexo_ws_schema::CronListResponse { jobs })
            }
            Err(e) => internal_error(request_id, format!("Failed to list cron jobs: {e}")),
        },
        Method::CronDelete => {
            let del_params: CronDeleteParams = match parse_params(request_id, params, "cron.delete") {
                Ok(p) => p,
                Err(f) => return f,
            };

            match crate::agent::cron::delete_job(db, &del_params.job_id).await {
                Ok(deleted) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::CronDeleteResponse { deleted },
                ),
                Err(e) => internal_error(request_id, format!("Failed to delete cron job: {e}")),
            }
        }
        Method::SystemPresence => {
            ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
        }
        Method::Send => {
            ok_or_internal_error(request_id, nexo_ws_schema::SendResponse { delivered: true })
        }
        Method::ModelStatus => {
            let status_params: ModelStatusParams = match parse_params(request_id, params, "model.status") {
                Ok(p) => p,
                Err(f) => return f,
            };
            let model_became_available = status_params.loaded_model_id.is_some();
            {
                let mut sw = state.write().await;
                sw.set_loaded_model(peer_id, status_params.loaded_model_id);
                sw.set_available_models(peer_id, status_params.available_models);
            }
            // Drain any queued runs now that a model is available
            if model_became_available {
                if let Err(e) = agent_handle.submit(AgentCommand::DrainQueue).await {
                    tracing::warn!("Failed to submit DrainQueue after ModelStatus: {e}");
                }
            }
            ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
        }
        Method::PrefillFetch => {
            let fetch_params: PrefillFetchParams = match parse_params(request_id, params, "prefill.fetch") {
                Ok(p) => p,
                Err(f) => return f,
            };
            let content = {
                let state_read = state.read().await;
                state_read.get_cached_prefill(&fetch_params.prefill_sha).map(String::from)
            };
            match content {
                Some(content) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::PrefillFetchResponse {
                        prefill_sha: fetch_params.prefill_sha,
                        content,
                    },
                ),
                None => Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "prefill_not_found".into(),
                        message: format!(
                            "Prefill SHA '{}' not found in cache",
                            fetch_params.prefill_sha
                        ),
                    },
                ),
            }
        }
        Method::PrefillMarkdownCreate => {
            let p: PrefillMarkdownCreateParams =
                match parse_params(request_id, params, "prefill.markdown.create") {
                    Ok(v) => v,
                    Err(f) => return f,
                };
            let storage_root = state.read().await.storage_root.clone();
            match crate::agent::prefill::create_markdown(
                db, &storage_root, &p.category, &p.description, &p.content,
            )
            .await
            {
                Ok(id) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::PrefillMarkdownCreateResponse { id },
                ),
                Err(e) => internal_error(request_id, format!("Failed to create markdown: {e}")),
            }
        }
        Method::PrefillMarkdownList => {
            match crate::agent::prefill::list_markdown(db).await {
                Ok(files) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::PrefillMarkdownListResponse {
                        files: files.into_iter().map(Into::into).collect(),
                    },
                ),
                Err(e) => internal_error(request_id, format!("Failed to list markdown files: {e}")),
            }
        }
        Method::PrefillMarkdownDelete => {
            let p: PrefillMarkdownDeleteParams =
                match parse_params(request_id, params, "prefill.markdown.delete") {
                    Ok(v) => v,
                    Err(f) => return f,
                };
            let storage_root = state.read().await.storage_root.clone();
            match crate::agent::prefill::delete_markdown(db, &storage_root, &p.id).await {
                Ok(deleted) => {
                    if deleted {
                        state.write().await.invalidate_prefill_cache();
                    }
                    ok_or_internal_error(
                        request_id,
                        nexo_ws_schema::PrefillMarkdownDeleteResponse { deleted },
                    )
                }
                Err(e) => internal_error(request_id, format!("Failed to delete markdown: {e}")),
            }
        }
        Method::PrefillCollectionCreate => {
            let p: PrefillCollectionCreateParams =
                match parse_params(request_id, params, "prefill.collection.create") {
                    Ok(v) => v,
                    Err(f) => return f,
                };
            match crate::agent::prefill::create_collection(
                db,
                &p.name,
                p.description.as_deref(),
                &p.markdown_ids,
            )
            .await
            {
                Ok(id) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::PrefillCollectionCreateResponse { id },
                ),
                Err(e) => internal_error(request_id, format!("Failed to create collection: {e}")),
            }
        }
        Method::PrefillCollectionList => {
            match crate::agent::prefill::list_collections(db).await {
                Ok(cols) => ok_or_internal_error(
                    request_id,
                    nexo_ws_schema::PrefillCollectionListResponse {
                        collections: cols.into_iter().map(Into::into).collect(),
                    },
                ),
                Err(e) => internal_error(request_id, format!("Failed to list collections: {e}")),
            }
        }
        Method::PrefillCollectionDelete => {
            let p: PrefillCollectionDeleteParams =
                match parse_params(request_id, params, "prefill.collection.delete") {
                    Ok(v) => v,
                    Err(f) => return f,
                };
            match crate::agent::prefill::delete_collection(db, &p.id).await {
                Ok(deleted) => {
                    if deleted {
                        state.write().await.invalidate_prefill_cache();
                    }
                    ok_or_internal_error(
                        request_id,
                        nexo_ws_schema::PrefillCollectionDeleteResponse { deleted },
                    )
                }
                Err(e) => internal_error(request_id, format!("Failed to delete collection: {e}")),
            }
        }
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    use crate::server::state::{GatewayState, dummy_sender};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_state() -> SharedState {
        Arc::new(RwLock::new(GatewayState::new(std::path::PathBuf::from("/tmp"))))
    }

    fn make_agent_handle(state: &SharedState, db: &SqlitePool) -> AgentHandle {
        let event_tx = {
            let st = state.try_read().unwrap();
            st.event_tx.clone()
        };
        AgentHandle::spawn(db.clone(), state.clone(), event_tx)
    }

    // Helper: dispatch with a real DB pool
    async fn dispatch(
        req_id: &str,
        method: &Method,
        params: serde_json::Value,
        peer_id: &str,
        state: &SharedState,
        db: &SqlitePool,
        agent_handle: &AgentHandle,
    ) -> Frame {
        dispatch_method(req_id, method, params, peer_id, state, db, agent_handle).await
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_health_returns_ok(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
        let resp = dispatch("req-1", &Method::Health, serde_json::json!({}), "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert_eq!(p["status"], "ok");
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_status_returns_counts(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
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
        let resp = dispatch("req-1", &Method::Status, serde_json::json!({}), "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert_eq!(p["connectedUsers"], 1);
            assert_eq!(p["connectedNodes"], 0);
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_connect_after_handshake_rejected(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
        let resp = dispatch("req-1", &Method::Connect, serde_json::json!({}), "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "invalid_method");
        } else {
            panic!("Expected error response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_tools_register_from_node(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
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

        let resp = dispatch("req-1", &Method::ToolsRegister, params, "n1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            assert_eq!(payload.unwrap()["registered"], 1);
        } else {
            panic!("Expected response");
        }

        // Verify tool is in catalog
        let catalog_resp = dispatch("req-2", &Method::ToolsCatalog, serde_json::json!({}), "n1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = catalog_resp {
            assert!(ok);
            let tools = &payload.unwrap()["tools"];
            assert_eq!(tools.as_array().unwrap().len(), 1);
            assert_eq!(tools[0]["name"], "echo.run");
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_tools_register_from_user_rejected(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
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
        let resp = dispatch("req-1", &Method::ToolsRegister, params, "u1", &state, &pool, &ah).await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "forbidden");
        } else {
            panic!("Expected error response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_tools_execute_tool_not_found(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
        let params = serde_json::json!({
            "tool": "nonexistent",
            "args": {},
            "idempotencyKey": "k1"
        });
        let resp = dispatch("req-1", &Method::ToolsExecute, params, "u1", &state, &pool, &ah).await;
        if let Frame::Response { ok, error, .. } = resp {
            assert!(!ok);
            assert_eq!(error.unwrap().code, "tool_not_found");
        } else {
            panic!("Expected error response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_session_create_returns_id(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);

        // Set up user FK
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool).await.unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
            .execute(&pool).await.unwrap();

        // Add peer so user_id_for_peer resolves
        {
            let mut s = state.write().await;
            s.add_peer(PeerInfo {
                id: "p1".into(), client_id: "cli".into(), role: Role::User,
                scopes: vec![], capabilities: vec![], commands: vec![],
                device_id: None, connected_at: chrono::Utc::now(),
            }, dummy_sender());
        }

        let params = serde_json::json!({"name": "test session"});
        let resp = dispatch("req-1", &Method::SessionCreate, params, "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert!(p["sessionId"].as_str().is_some());
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_session_list_empty(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);
        {
            let mut s = state.write().await;
            s.add_peer(PeerInfo {
                id: "p1".into(), client_id: "cli".into(), role: Role::User,
                scopes: vec![], capabilities: vec![], commands: vec![],
                device_id: None, connected_at: chrono::Utc::now(),
            }, dummy_sender());
        }

        let resp = dispatch("req-1", &Method::SessionList, serde_json::json!({}), "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let sessions = payload.unwrap()["sessions"].as_array().unwrap().clone();
            assert!(sessions.is_empty());
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_agent_returns_accepted_with_session(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);

        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(&pool).await.unwrap();
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
            .execute(&pool).await.unwrap();

        {
            let mut s = state.write().await;
            s.add_peer(PeerInfo {
                id: "p1".into(), client_id: "cli".into(), role: Role::User,
                scopes: vec![], capabilities: vec![], commands: vec![],
                device_id: None, connected_at: chrono::Utc::now(),
            }, dummy_sender());
        }

        let params = serde_json::json!({
            "prompt": "hello",
            "idempotencyKey": "k1"
        });
        let resp = dispatch("req-1", &Method::Agent, params, "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            let p = payload.unwrap();
            assert_eq!(p["status"], "accepted");
            assert!(p["runId"].as_str().is_some());
            assert!(p["sessionId"].as_str().is_some());
        } else {
            panic!("Expected response");
        }
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn dispatch_cron_create_and_list(pool: SqlitePool) {
        let state = make_state();
        let ah = make_agent_handle(&state, &pool);

        let params = serde_json::json!({
            "name": "test job",
            "schedule": "0 * * * *",
            "prompt": "hello"
        });
        let resp = dispatch("req-1", &Method::CronCreate, params, "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = resp {
            assert!(ok);
            assert!(payload.unwrap()["jobId"].as_str().is_some());
        } else {
            panic!("Expected response");
        }

        let list_resp = dispatch("req-2", &Method::CronList, serde_json::json!({}), "p1", &state, &pool, &ah).await;
        if let Frame::Response { ok, payload, .. } = list_resp {
            assert!(ok);
            let jobs = payload.unwrap()["jobs"].as_array().unwrap().clone();
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0]["name"], "test job");
        } else {
            panic!("Expected response");
        }
    }
}
