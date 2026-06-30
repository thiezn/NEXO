use crate::Result;
use futures_util::{SinkExt, StreamExt};
use nexo_core::{GatewayProperties, NexoClient, NexoClientKind, OperationId};
use nexo_ws_schema::{
    ConnectRequest, Frame, GatewayToNodeMessage, GatewayToUserMessage, NexoResponse,
    NodeToGatewayMessage, UserToGatewayMessage,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info};
use uuid::Uuid;

/// Unique live peer key derived from stable client and device identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerKey {
    client_id: Uuid,
    device_id: Uuid,
}

impl PeerKey {
    /// Build a peer key from stable client and device identifiers.
    ///
    /// # Arguments
    ///
    /// * `client_id` - Stable client identifier advertised by the peer.
    /// * `device_id` - Stable device identifier advertised by the peer.
    pub fn new(client_id: Uuid, device_id: Uuid) -> Self {
        Self {
            client_id,
            device_id,
        }
    }

    /// Build a peer key from a connected Nexo client payload.
    ///
    /// # Arguments
    ///
    /// * `client` - The domain-level client payload received during connect.
    pub fn from_client(client: &NexoClient) -> Self {
        Self::new(client.client().id, client.device().id)
    }

    /// Return the stable client identifier.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn client_id(&self) -> Uuid {
        self.client_id
    }

    /// Return the stable device identifier.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn device_id(&self) -> Uuid {
        self.device_id
    }
}

/// State owned by a single WebSocket connection task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GatewayConnectionState {
    /// The connection has not completed the Nexo connect handshake yet.
    AwaitingConnect,

    /// The connection is bound to a Nexo peer.
    Connected {
        /// Stable live peer key for this connection.
        peer_key: PeerKey,

        /// Domain-level client kind bound to this connection.
        kind: NexoClientKind,
    },

    /// The connection completed a graceful protocol disconnect.
    Disconnected,
}

/// Result of handling one inbound gateway frame.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayFrameOutcome {
    /// Send a reply frame and keep the connection open.
    Reply(Frame),

    /// Send a reply frame, then close the connection.
    CloseAfterReply(Frame),

    /// The frame parsed successfully but no reply is produced in this initial slice.
    NoReply,
}

/// Small private helper to be able to extract the NexoClientKind during initial
/// connect, before the connection state is bound to a peer key.
#[derive(Debug, serde::Deserialize)]
enum GatewayConnectMessage {
    Connect(ConnectRequest),
}

/// Central coordinator for nexo-gateway, ties configuration,
/// tool registry, websocket loop and inference engine together.
#[derive(Clone)]
pub struct NexoGateway {
    /// The configuration for the gateway.
    config: GatewayProperties,

    /// Connected peers by stable `(client_id, device_id)` key.
    peers: Arc<Mutex<HashMap<PeerKey, NexoClient>>>,
}

impl NexoGateway {
    /// Create a gateway coordinator from prepared gateway properties.
    ///
    /// # Arguments
    ///
    /// * `config` - Gateway runtime configuration, including bind address and auth token.
    pub fn new(config: GatewayProperties) -> Self {
        Self {
            config,
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start the gateway runtime.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub async fn run(&self) -> Result {
        let addr = self.config.bind_addr();
        let listener = TcpListener::bind(&addr).await?;
        info!(addr = %addr, "NEXO Gateway listening");

        loop {
            let (stream, peer_addr) = listener.accept().await?;
            let gateway = self.clone();
            let auth_token = self.config.auth_token().to_owned();

            tokio::spawn(async move {
                #[allow(clippy::result_large_err)]
                let callback = |req: &http::Request<()>, response: http::Response<()>| {
                    if has_valid_auth(req.headers(), &auth_token) {
                        Ok(response)
                    } else {
                        let mut response = http::Response::new(Some("Unauthorized".to_owned()));
                        *response.status_mut() = http::StatusCode::UNAUTHORIZED;
                        Err(response)
                    }
                };

                match tokio_tungstenite::accept_hdr_async(stream, callback).await {
                    Ok(ws_stream) => gateway.handle_connection(ws_stream).await,
                    Err(error) => {
                        tracing::warn!(peer_addr = %peer_addr, error = ?error, "WebSocket handshake failed");
                    }
                }
            });
        }
    }

    /// Handle a frame received on a single gateway WebSocket connection.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection protocol state for the socket that received the frame.
    /// * `frame` - The incoming gateway frame to parse and handle.
    fn handle_frame(
        &self,
        state: &mut GatewayConnectionState,
        frame: Frame,
    ) -> Result<GatewayFrameOutcome> {
        match state {
            GatewayConnectionState::AwaitingConnect => self.handle_awaiting_connect(state, frame),
            GatewayConnectionState::Connected { kind, .. } => match kind {
                NexoClientKind::User => {
                    let (_, message) = frame.into_parts::<UserToGatewayMessage>()?;
                    self.handle_user_message(state, message)
                }
                NexoClientKind::Node => {
                    let (_, message) = frame.into_parts::<NodeToGatewayMessage>()?;
                    self.handle_node_message(state, message)
                }
            },
            GatewayConnectionState::Disconnected => Err(crate::Error::InvalidPeerState(
                "frame received after protocol disconnect".into(),
            )),
        }
    }

    /// Return the connected client for a live peer key.
    ///
    /// # Arguments
    ///
    /// * `peer_key` - Stable `(client_id, device_id)` key for the connected peer.
    pub fn peer(&self, peer_key: PeerKey) -> Option<NexoClient> {
        self.peers
            .lock()
            .ok()
            .and_then(|peers| peers.get(&peer_key).cloned())
    }

    /// Handle the first application frame for a socket that has not connected yet.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state to bind after a successful connect.
    /// * `frame` - Incoming frame expected to contain a connect request.
    fn handle_awaiting_connect(
        &self,
        state: &mut GatewayConnectionState,
        frame: Frame,
    ) -> Result<GatewayFrameOutcome> {
        match frame.into_parts::<GatewayConnectMessage>() {
            Ok((_, GatewayConnectMessage::Connect(request))) => self.connect_peer(state, request),
            Err(_) => Err(crate::Error::InvalidPeerState(
                "first gateway frame must be a connect request".into(),
            )),
        }
    }

    /// Register a connected peer and build the matching connect response frame.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state to transition to connected.
    /// * `request` - Parsed connect request containing the connecting `NexoClient`.
    fn connect_peer(
        &self,
        state: &mut GatewayConnectionState,
        request: ConnectRequest,
    ) -> Result<GatewayFrameOutcome> {
        let peer_key = PeerKey::from_client(&request.client);
        let kind = request.client.kind();
        let client_id = request.client.client().id;
        let device_id = request.client.device().id;
        self.peers
            .lock()
            .map_err(|_| crate::Error::InvalidPeerState("peer state lock poisoned".into()))?
            .insert(peer_key, request.client);
        *state = GatewayConnectionState::Connected { peer_key, kind };

        info!(kind = %kind, client_id = %client_id, device_id = %device_id, "Peer connected");

        Ok(match kind {
            NexoClientKind::User => GatewayFrameOutcome::Reply(Frame::new(
                GatewayToUserMessage::Connect(completed_response(request.operation_id)),
            )?),
            NexoClientKind::Node => GatewayFrameOutcome::Reply(Frame::new(
                GatewayToNodeMessage::Connect(completed_response(request.operation_id)),
            )?),
        })
    }

    /// Handle a message received from a connected user peer.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state for the user socket.
    /// * `message` - Parsed user-to-gateway protocol message.
    fn handle_user_message(
        &self,
        state: &mut GatewayConnectionState,
        message: UserToGatewayMessage,
    ) -> Result<GatewayFrameOutcome> {
        match message {
            UserToGatewayMessage::Connect(_) => Err(crate::Error::InvalidPeerState(
                "connect received after peer was already connected".into(),
            )),
            UserToGatewayMessage::Disconnect(request) => {
                self.disconnect_peer(state)?;
                Ok(GatewayFrameOutcome::CloseAfterReply(Frame::new(
                    GatewayToUserMessage::Disconnect(completed_response(request.operation_id)),
                )?))
            }
            other => {
                let name: &'static str = (&other).into();
                debug!(message = name, "User message parsed for later routing");
                Ok(GatewayFrameOutcome::NoReply)
            }
        }
    }

    /// Handle a message received from a connected node peer.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state for the node socket.
    /// * `message` - Parsed node-to-gateway protocol message.
    fn handle_node_message(
        &self,
        state: &mut GatewayConnectionState,
        message: NodeToGatewayMessage,
    ) -> Result<GatewayFrameOutcome> {
        match message {
            NodeToGatewayMessage::Connect(_) => Err(crate::Error::InvalidPeerState(
                "connect received after peer was already connected".into(),
            )),
            NodeToGatewayMessage::Disconnect(request) => {
                self.disconnect_peer(state)?;
                Ok(GatewayFrameOutcome::CloseAfterReply(Frame::new(
                    GatewayToNodeMessage::Disconnect(completed_response(request.operation_id)),
                )?))
            }
            other => {
                let name: &'static str = (&other).into();
                debug!(message = name, "Node message parsed for later routing");
                Ok(GatewayFrameOutcome::NoReply)
            }
        }
    }

    /// Remove a connected peer from live gateway state.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state that identifies the peer to remove.
    fn disconnect_peer(&self, state: &mut GatewayConnectionState) -> Result {
        let GatewayConnectionState::Connected { peer_key, kind } = state else {
            return Err(crate::Error::InvalidPeerState(
                "disconnect received before peer was connected".into(),
            ));
        };

        if self
            .peers
            .lock()
            .map_err(|_| crate::Error::InvalidPeerState("peer state lock poisoned".into()))?
            .remove(peer_key)
            .is_some()
        {
            info!(kind = %kind, client_id = %peer_key.client_id(), device_id = %peer_key.device_id(), "Peer disconnected");
        }
        *state = GatewayConnectionState::Disconnected;
        Ok(())
    }

    /// Run the read/write loop for one accepted WebSocket connection.
    ///
    /// # Arguments
    ///
    /// * `ws_stream` - Accepted WebSocket stream for one connected peer.
    async fn handle_connection(&self, mut ws_stream: WebSocketStream<TcpStream>) {
        let mut state = GatewayConnectionState::AwaitingConnect;

        while let Some(message) = ws_stream.next().await {
            let frame = match message {
                Ok(Message::Text(text)) => match serde_json::from_str::<Frame>(&text) {
                    Ok(frame) => frame,
                    Err(error) => {
                        tracing::warn!(error = ?error, "Received invalid frame JSON");
                        break;
                    }
                },
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => continue,
                Ok(Message::Binary(_)) => {
                    tracing::warn!("Received unexpected binary WebSocket message");
                    continue;
                }
            };

            match self.handle_frame(&mut state, frame) {
                Ok(GatewayFrameOutcome::Reply(frame)) => {
                    if let Err(error) = send_frame(&mut ws_stream, &frame).await {
                        tracing::warn!(error = ?error, "Failed to send gateway reply");
                        break;
                    }
                }
                Ok(GatewayFrameOutcome::CloseAfterReply(frame)) => {
                    if let Err(error) = send_frame(&mut ws_stream, &frame).await {
                        tracing::warn!(error = ?error, "Failed to send gateway disconnect reply");
                    }
                    let _ = ws_stream.close(None).await;
                    return;
                }
                Ok(GatewayFrameOutcome::NoReply) => {}
                Err(error) => {
                    tracing::warn!(error = ?error, "Gateway frame handling failed");
                    break;
                }
            }
        }

        self.cleanup_connection(&mut state);
    }

    /// Remove a connected peer when its socket closes without a protocol disconnect.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state to inspect and clean up.
    fn cleanup_connection(&self, state: &mut GatewayConnectionState) {
        if matches!(state, GatewayConnectionState::Connected { .. })
            && let Err(error) = self.disconnect_peer(state)
        {
            tracing::warn!(error = ?error, "Failed to clean up peer connection");
        }
    }
}

/// Build a successful response for an operation.
///
/// # Arguments
///
/// * `operation_id` - Operation identifier from the request being completed.
fn completed_response(operation_id: OperationId) -> NexoResponse {
    NexoResponse::Completed {
        operation_id,
        result: (),
    }
}

/// Check whether a WebSocket handshake carries the expected gateway auth header.
///
/// # Arguments
///
/// * `headers` - HTTP headers from the WebSocket upgrade request.
/// * `expected_token` - Auth token configured for the gateway.
fn has_valid_auth(headers: &http::HeaderMap, expected_token: &str) -> bool {
    headers
        .get(nexo_ws_schema::AUTH_HEADER)
        .and_then(|value| value.to_str().ok())
        == Some(expected_token)
}

/// Send a serialized gateway frame over a WebSocket stream.
///
/// # Arguments
///
/// * `ws_stream` - WebSocket stream to write the frame to.
/// * `frame` - Frame envelope to serialize as a text message.\
#[inline]
async fn send_frame(ws_stream: &mut WebSocketStream<TcpStream>, frame: &Frame) -> Result {
    let json = serde_json::to_string(frame)?;
    ws_stream.send(Message::Text(json.into())).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, GatewayProperties, NodeProperties, UserProperties};
    use nexo_ws_client::NexoConnection;
    use nexo_ws_schema::{DisconnectRequest, NexoResponse};

    fn gateway() -> NexoGateway {
        NexoGateway::new(GatewayProperties::default())
    }

    fn user_client() -> NexoClient {
        NexoClient::User(UserProperties::new(
            ClientInfo::new("test-user"),
            DeviceInfo::default(),
            "token",
        ))
    }

    fn node_client() -> NexoClient {
        NexoClient::Node(NodeProperties::new(
            ClientInfo::new("test-node"),
            DeviceInfo::default(),
            "token",
        ))
    }

    fn operation_id_from_user_connect(frame: &Frame) -> OperationId {
        let (_, message) = frame.clone().into_parts::<GatewayToUserMessage>().unwrap();
        let GatewayToUserMessage::Connect(NexoResponse::Completed { operation_id, .. }) = message
        else {
            panic!("expected user connect completed response")
        };
        operation_id
    }

    #[test]
    fn user_connect_binds_peer_and_preserves_operation_id() {
        let gateway = gateway();
        let mut state = GatewayConnectionState::AwaitingConnect;
        let request = ConnectRequest::new(user_client());
        let expected_operation_id = request.operation_id;
        let frame = Frame::new(UserToGatewayMessage::Connect(request)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).unwrap();

        let GatewayFrameOutcome::Reply(reply) = outcome else {
            panic!("expected connect reply")
        };
        assert_eq!(
            operation_id_from_user_connect(&reply),
            expected_operation_id
        );
        let GatewayConnectionState::Connected { peer_key, kind } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::User);
        assert_eq!(gateway.peer(peer_key).unwrap().kind(), NexoClientKind::User);
    }

    #[test]
    fn node_connect_binds_node_kind() {
        let gateway = gateway();
        let mut state = GatewayConnectionState::AwaitingConnect;
        let request = ConnectRequest::new(node_client());
        let frame = Frame::new(NodeToGatewayMessage::Connect(request)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).unwrap();

        assert!(matches!(outcome, GatewayFrameOutcome::Reply(_)));
        let GatewayConnectionState::Connected { peer_key, kind } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::Node);
        assert_eq!(gateway.peer(peer_key).unwrap().kind(), NexoClientKind::Node);
    }

    #[test]
    fn first_connect_is_classified_by_nexo_client_kind() {
        let gateway = gateway();
        let mut state = GatewayConnectionState::AwaitingConnect;
        let frame = Frame::new(UserToGatewayMessage::Connect(ConnectRequest::new(
            node_client(),
        )))
        .unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).unwrap();

        assert!(matches!(outcome, GatewayFrameOutcome::Reply(_)));
        let GatewayConnectionState::Connected { kind, .. } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::Node);
    }

    #[test]
    fn disconnect_removes_peer_and_closes_after_reply() {
        let gateway = gateway();
        let mut state = GatewayConnectionState::AwaitingConnect;
        let connect_frame = Frame::new(UserToGatewayMessage::Connect(ConnectRequest::new(
            user_client(),
        )))
        .unwrap();
        gateway.handle_frame(&mut state, connect_frame).unwrap();
        let GatewayConnectionState::Connected { peer_key, .. } = state else {
            panic!("expected connected state")
        };
        assert!(gateway.peer(peer_key).is_some());
        let disconnect = DisconnectRequest::new();
        let expected_operation_id = disconnect.operation_id;
        let frame = Frame::new(UserToGatewayMessage::Disconnect(disconnect)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).unwrap();

        let GatewayFrameOutcome::CloseAfterReply(reply) = outcome else {
            panic!("expected close-after-reply")
        };
        let (_, message) = reply.into_parts::<GatewayToUserMessage>().unwrap();
        let GatewayToUserMessage::Disconnect(NexoResponse::Completed { operation_id, .. }) =
            message
        else {
            panic!("expected user disconnect completed response")
        };
        assert_eq!(operation_id, expected_operation_id);
        assert_eq!(state, GatewayConnectionState::Disconnected);
        assert!(gateway.peer(peer_key).is_none());
    }

    #[test]
    fn first_frame_must_be_connect() {
        let gateway = gateway();
        let mut state = GatewayConnectionState::AwaitingConnect;
        let frame = Frame::new(UserToGatewayMessage::Disconnect(DisconnectRequest::new())).unwrap();

        let error = gateway.handle_frame(&mut state, frame).unwrap_err();

        assert!(matches!(error, crate::Error::InvalidPeerState(_)));
        assert_eq!(state, GatewayConnectionState::AwaitingConnect);
    }

    #[tokio::test]
    async fn run_accepts_user_connect_and_disconnect() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let gateway = gateway();
        let server = gateway.clone();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let callback = |req: &http::Request<()>, response: http::Response<()>| {
                if has_valid_auth(req.headers(), "token") {
                    Ok(response)
                } else {
                    let mut response = http::Response::new(Some("Unauthorized".to_owned()));
                    *response.status_mut() = http::StatusCode::UNAUTHORIZED;
                    Err(response)
                }
            };
            let ws_stream = tokio_tungstenite::accept_hdr_async(stream, callback)
                .await
                .unwrap();
            server.handle_connection(ws_stream).await;
        });

        let user =
            UserProperties::builder(ClientInfo::new("test-user"), DeviceInfo::default(), "token")
                .gateway_url(format!("ws://{addr}"))
                .build();
        let mut conn = NexoConnection::connect(user.gateway_url(), NexoClient::User(user.clone()))
            .await
            .unwrap();

        let key = PeerKey::new(user.client().id, user.device().id);
        assert_eq!(gateway.peer(key).unwrap().kind(), NexoClientKind::User);

        let disconnect = DisconnectRequest::new();
        let expected_operation_id = disconnect.operation_id;
        let frame = Frame::new(UserToGatewayMessage::Disconnect(disconnect)).unwrap();
        conn.send_frame(&frame).await.unwrap();
        let reply = conn.recv_frame().await.unwrap();
        let (_, message) = reply.into_parts::<GatewayToUserMessage>().unwrap();
        let GatewayToUserMessage::Disconnect(NexoResponse::Completed { operation_id, .. }) =
            message
        else {
            panic!("expected disconnect response")
        };
        assert_eq!(operation_id, expected_operation_id);

        server_task.await.unwrap();
        assert!(gateway.peer(key).is_none());
    }
}
