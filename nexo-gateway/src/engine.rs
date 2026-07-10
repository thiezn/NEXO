use super::agent::{NexoAgentInput, NexoAgentOutput};
use crate::{Error, NexoAgent, Result};
use futures_util::{SinkExt, StreamExt};
use nexo_core::system::node::NodeState;
use nexo_core::{GatewayProperties, NexoClient, NexoClientKind, Node, PeerId, User};
use nexo_ws_schema::{
    ConnectRequest, Frame, GatewayToNodeMessage, GatewayToUserMessage, NexoResponse,
    NodeToGatewayMessage, UserToGatewayMessage,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use strum::EnumDiscriminants;
use tracing::{debug, info, warn};

const AGENT_CHANNEL_CAPACITY: usize = 256;
const PEER_CHANNEL_CAPACITY: usize = 64;

/// State owned by a single WebSocket connection task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumDiscriminants)]
#[strum_discriminants(name(PeerConnectionStateKind))]
#[strum_discriminants(vis(pub(crate)))]
#[strum_discriminants(doc = "The persisted connection-state kind of a gateway peer." )]
#[strum_discriminants(derive(
    Hash,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr
))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
enum PeerConnectionState {
    /// The connection has not completed the Nexo connect handshake yet.
    AwaitingConnect,

    /// The connection is bound to a Nexo peer.
    Connected {
        /// Stable live peer key for this connection.
        peer_id: PeerId,

        /// Domain-level client kind bound to this connection.
        kind: NexoClientKind,
    },

    /// The connection completed a graceful protocol disconnect.
    Disconnected,
}

/// Result of handling one inbound gateway frame.
#[derive(Debug, Clone, PartialEq)]
enum GatewayFrameOutput {
    /// Send a reply frame and keep the connection open.
    Reply(Frame),

    /// Send a reply frame, then close the connection.
    CloseAfterReply(Frame),

    /// The frame parsed successfully but no reply is produced in this initial slice.
    NoReply,
}

/// Small private helper to be able to extract the NexoClientKind during initial
/// connect, before the connection state is bound to a peer key.
///
/// It mirrors the GatewayToNode/User message enum, but only contains the Connect variant, allowing
/// us to have generic handling of the connect request without needing to know the type of client.
///
/// TODO: We could probably fix this better my splitting out the Connect messages into separate enum?
#[derive(Debug, serde::Deserialize)]
enum GatewayConnectMessage {
    Connect(ConnectRequest),
}

/// Central coordinator for nexo-gateway, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub struct NexoGateway {
    /// The configuration for the gateway.
    config: GatewayProperties,

    /// Connected peers by stable `(client_id, device_id)` key.
    peers: Arc<Mutex<HashMap<PeerId, NexoClient>>>,

    /// Sender used by gateway connection tasks to forward inputs into the NexoAgent.
    agent_input_tx: mpsc::Sender<NexoAgentInput>,

    /// One-time receiver consumed when the NexoAgent runtime is started.
    agent_input_rx: Option<mpsc::Receiver<NexoAgentInput>>,

    /// Sender used by the NexoAgent to emit outputs back into the gateway runtime.
    agent_output_tx: mpsc::Sender<NexoAgentOutput>,

    /// One-time receiver consumed by the gateway output dispatcher task.
    agent_output_rx: Option<mpsc::Receiver<NexoAgentOutput>>,

    /// Per-peer directed channels used to push outbound frames to the right websocket task.
    peer_frame_txs: Arc<Mutex<HashMap<PeerId, mpsc::Sender<Frame>>>>,
}

impl Clone for NexoGateway {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            peers: Arc::clone(&self.peers),
            agent_input_tx: self.agent_input_tx.clone(),
            agent_input_rx: None,
            agent_output_tx: self.agent_output_tx.clone(),
            agent_output_rx: None,
            peer_frame_txs: Arc::clone(&self.peer_frame_txs),
        }
    }
}

impl NexoGateway {
    /// Create a gateway coordinator from prepared gateway properties.
    ///
    /// # Arguments
    ///
    /// * `config` - Gateway runtime configuration, including bind address and auth token.
    pub fn new(config: GatewayProperties) -> Result<Self> {
        let (agent_input_tx, agent_input_rx) = mpsc::channel(AGENT_CHANNEL_CAPACITY);
        let (agent_output_tx, agent_output_rx) = mpsc::channel(AGENT_CHANNEL_CAPACITY);

        Ok(Self {
            config,
            peers: Arc::new(Mutex::new(HashMap::new())),
            agent_input_tx,
            agent_input_rx: Some(agent_input_rx),
            agent_output_tx,
            agent_output_rx: Some(agent_output_rx),
            peer_frame_txs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Start the gateway runtime.
    ///
    /// This will consume NexoGateway and run the main accept loop,
    /// spawning a new task for each accepted WebSocket connection.
    pub async fn run(mut self) -> Result {
        let agent_input_rx = self
            .agent_input_rx
            .take()
            .ok_or_else(|| Error::InvalidPeerState("agent input receiver already taken".into()))?;
        let agent_output_rx = self
            .agent_output_rx
            .take()
            .ok_or_else(|| Error::InvalidPeerState("agent output receiver already taken".into()))?;

        let mut agent_task = NexoAgent::new().start(agent_input_rx, self.agent_output_tx.clone());
        let mut dispatcher_task = self.start_agent_output_dispatcher(agent_output_rx);

        let addr = self.config.bind_addr();
        let listener = TcpListener::bind(&addr).await?;
        info!(addr = %addr, "NEXO Gateway listening");

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    debug!(accept_result = ?accept_result, "TCP connection accepted");
                    let (stream, peer_addr) = accept_result?;
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
                agent_result = &mut agent_task => {
                    agent_result??;
                    return Ok(());
                }
                dispatcher_result = &mut dispatcher_task => {
                    dispatcher_result??;
                    return Ok(());
                }
            }
        }
    }

    /// Handle a frame received on a single gateway WebSocket connection.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection protocol state for the socket that received the frame.
    /// * `frame` - The incoming gateway frame to parse and handle.
    async fn handle_frame(
        &self,
        state: &mut PeerConnectionState,
        frame: Frame,
    ) -> Result<GatewayFrameOutput> {
        match state {
            PeerConnectionState::AwaitingConnect => self.handle_awaiting_connect(state, frame),
            PeerConnectionState::Connected { kind, .. } => match kind {
                NexoClientKind::User => {
                    debug!(frame = ?frame, "User frame received");
                    let (_, message) = frame.into_parts::<UserToGatewayMessage>()?;
                    self.handle_user_message(state, message).await
                }
                NexoClientKind::Node => {
                    debug!(frame = ?frame, "Node frame received");
                    let (_, message) = frame.into_parts::<NodeToGatewayMessage>()?;
                    self.handle_node_message(state, message)
                }
            },
            PeerConnectionState::Disconnected => Err(Error::InvalidPeerState(
                "frame received after protocol disconnect".into(),
            )),
        }
    }

    /// Return the connected client for a live peer key.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Stable `(client_id, device_id)` key for the connected peer.
    pub fn peer(&self, peer_id: PeerId) -> Option<NexoClient> {
        self.peers
            .lock()
            .ok()
            .and_then(|peers| peers.get(&peer_id).cloned())
    }

    /// Handle the first application frame for a socket that has not connected yet.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state to bind after a successful connect.
    /// * `frame` - Incoming frame expected to contain a connect request.
    fn handle_awaiting_connect(
        &self,
        state: &mut PeerConnectionState,
        frame: Frame,
    ) -> Result<GatewayFrameOutput> {
        match frame.into_parts::<GatewayConnectMessage>() {
            Ok((_, GatewayConnectMessage::Connect(request))) => self.connect_peer(state, request),
            Err(_) => Err(Error::InvalidPeerState(
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
        state: &mut PeerConnectionState,
        request: ConnectRequest,
    ) -> Result<GatewayFrameOutput> {
        let peer_id = PeerId::from_client(&request.client);

        let kind = request.client.kind();
        self.peers
            .lock()
            .map_err(|_| Error::InvalidPeerState("peer state lock poisoned".into()))?
            .insert(peer_id, request.client.clone());

        *state = PeerConnectionState::Connected { peer_id, kind };

        if let Err(error) = self.agent_input_tx.try_send(match &request.client {
            NexoClient::Node(properties) => NexoAgentInput::NodeConnected(Node::from_properties(
                properties,
                NodeState::Idle,
                std::collections::HashSet::new(),
            )),
            NexoClient::User(properties) => {
                NexoAgentInput::UserConnected(User::from_properties(properties))
            }
        }) {
            warn!(error = ?error, peer_id = %peer_id, "Failed to forward connect event to agent");
        }

        Ok(match request.client {
            NexoClient::Node(properties) => {
                info!(client_id = %properties.client().id, device_id = %properties.device().id, tools = ?properties.tools(), "Node peer connected");

                GatewayFrameOutput::Reply(Frame::new(GatewayToNodeMessage::Connect(
                    NexoResponse::completed(request.operation_id),
                ))?)
            }
            NexoClient::User(properties) => {
                info!(client_id = %properties.client().id, device_id = %properties.device().id, "User peer connected");
                GatewayFrameOutput::Reply(Frame::new(GatewayToUserMessage::Connect(
                    NexoResponse::completed(request.operation_id),
                ))?)
            }
        })
    }

    /// Handle a message received from a connected user peer.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable per-connection state for the user socket.
    /// * `message` - Parsed user-to-gateway protocol message.
    async fn handle_user_message(
        &self,
        state: &mut PeerConnectionState,
        message: UserToGatewayMessage,
    ) -> Result<GatewayFrameOutput> {
        match message {
            UserToGatewayMessage::Connect(_) => Err(Error::InvalidPeerState(
                "connect received after peer was already connected".into(),
            )),
            UserToGatewayMessage::Disconnect(request) => {
                self.disconnect_peer(state)?;
                Ok(GatewayFrameOutput::CloseAfterReply(Frame::new(
                    GatewayToUserMessage::Disconnect(NexoResponse::completed(request.operation_id)),
                )?))
            }
            UserToGatewayMessage::StartInferenceRun(intent) => {
                let PeerConnectionState::Connected { peer_id, .. } = state else {
                    return Err(Error::InvalidPeerState(
                        "start_inference_run received before peer was connected".into(),
                    ));
                };

                if let Err(error) = self
                    .agent_input_tx
                    .try_send(NexoAgentInput::UserStartInferenceRun {
                        requester: *peer_id,
                        intent: intent,
                    })
                {
                    warn!(error = %error, "Failed to forward StartInferenceRun to agent");
                }
                Ok(GatewayFrameOutput::NoReply)
            }
            UserToGatewayMessage::GetState => {
                let PeerConnectionState::Connected { peer_id, .. } = state else {
                    return Err(Error::InvalidPeerState(
                        "get_state received before peer was connected".into(),
                    ));
                };

                let operation_id = nexo_core::OperationId::new();
                self.agent_input_tx.try_send(NexoAgentInput::GetState {
                    requester: *peer_id,
                    operation_id,
                })?;

                Ok(GatewayFrameOutput::NoReply)
            }
            other => {
                // let name: &'static str = (&other).into();
                info!(message = ?other, "User message parsed for later routing");
                Ok(GatewayFrameOutput::NoReply)
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
        state: &mut PeerConnectionState,
        message: NodeToGatewayMessage,
    ) -> Result<GatewayFrameOutput> {
        match message {
            NodeToGatewayMessage::Connect(_) => Err(Error::InvalidPeerState(
                "connect received after peer was already connected".into(),
            )),
            NodeToGatewayMessage::Disconnect(request) => {
                self.disconnect_peer(state)?;
                Ok(GatewayFrameOutput::CloseAfterReply(Frame::new(
                    GatewayToNodeMessage::Disconnect(NexoResponse::completed(request.operation_id)),
                )?))
            }
            other => {
                info!(message = ?other, "Node message parsed for later routing");
                Ok(GatewayFrameOutput::NoReply)
            }
        }
    }

    /// Handle a message received from the NexoAgent.
    fn handle_agent_output(&self, message: NexoAgentOutput) -> Result {
        match message {
            NexoAgentOutput::StartInferenceRun(_node, _request) => {
                info!("StartInferenceRun output received but routing is not implemented yet");
            }
            NexoAgentOutput::GetState {
                requester,
                operation_id,
                state,
            } => {
                let frame = Frame::new(GatewayToUserMessage::GetState(NexoResponse::Completed {
                    operation_id,
                    result: state,
                }))?;

                if let Some(tx) = self
                    .peer_frame_txs
                    .lock()
                    .map_err(|_| Error::InvalidPeerState("peer sender lock poisoned".into()))?
                    .get(&requester)
                    .cloned()
                {
                    tx.try_send(frame)?;
                } else {
                    info!(peer_id = %requester, "Dropping get_state response for disconnected peer");
                }
            }
        }

        Ok(())
    }

    /// Remove a connected peer from live gateway state.
    fn disconnect_peer(&self, state: &mut PeerConnectionState) -> Result {
        let PeerConnectionState::Connected { peer_id, kind } = state else {
            return Err(Error::InvalidPeerState(
                "disconnect received before peer was connected".into(),
            ));
        };

        if self
            .peers
            .lock()
            .map_err(|_| Error::InvalidPeerState("peer state lock poisoned".into()))?
            .remove(peer_id)
            .is_some()
        {
            info!(kind = %kind, client_id = %peer_id.client_id(), device_id = %peer_id.device_id(), "Peer disconnected");

            let input = match kind {
                NexoClientKind::User => NexoAgentInput::UserDisconnected(*peer_id),
                NexoClientKind::Node => NexoAgentInput::NodeDisconnected(*peer_id),
            };
            if let Err(error) = self.agent_input_tx.try_send(input) {
                warn!(error = %error, peer_id = %peer_id, "Failed to forward disconnect to agent");
            }
        }

        if let Ok(mut peer_senders) = self.peer_frame_txs.lock() {
            peer_senders.remove(peer_id);
        }

        *state = PeerConnectionState::Disconnected;
        Ok(())
    }

    /// Run the read/write loop for one accepted WebSocket connection.
    async fn handle_connection(&self, mut ws_stream: WebSocketStream<TcpStream>) {
        let mut state = PeerConnectionState::AwaitingConnect;
        let (peer_tx, mut peer_rx) = mpsc::channel::<Frame>(PEER_CHANNEL_CAPACITY);

        loop {
            tokio::select! {
                maybe_message = ws_stream.next() => {
                    let Some(message) = maybe_message else {
                        break;
                    };
                    debug!(message = ?message, "WebSocket message received");
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

                    match self.handle_frame(&mut state, frame).await {
                        Ok(GatewayFrameOutput::Reply(frame)) => {
                            debug!(frame = ?frame, "Sending gateway reply");
                            if let PeerConnectionState::Connected { peer_id, .. } = state {
                                self.register_peer_sender(peer_id, &peer_tx);
                            }

                            if let Err(error) = send_frame(&mut ws_stream, &frame).await {
                                tracing::warn!(error = ?error, "Failed to send gateway reply");
                                break;
                            }
                        }
                        Ok(GatewayFrameOutput::CloseAfterReply(frame)) => {
                            debug!(frame = ?frame, "Sending gateway disconnect reply");
                            if let Err(error) = send_frame(&mut ws_stream, &frame).await {
                                tracing::warn!(error = ?error, "Failed to send gateway disconnect reply");
                            }
                            let _ = ws_stream.close(None).await;
                            return;
                        }
                        Ok(GatewayFrameOutput::NoReply) => {}
                        Err(error) => {
                            tracing::warn!(error = ?error, "Gateway frame handling failed");
                            break;
                        }
                    }
                }
                Some(frame) = peer_rx.recv() => {
                    if let Err(error) = send_frame(&mut ws_stream, &frame).await {
                        tracing::warn!(error = ?error, "Failed to send directed peer frame");
                        break;
                    }
                }
            }
        }

        self.cleanup_connection(&mut state);
    }

    /// Remove a connected peer when its socket closes without a protocol disconnect.
    fn cleanup_connection(&self, state: &mut PeerConnectionState) {
        if matches!(state, PeerConnectionState::Connected { .. })
            && let Err(error) = self.disconnect_peer(state)
        {
            tracing::warn!(error = ?error, "Failed to clean up peer connection");
        }
    }

    fn register_peer_sender(&self, peer_id: PeerId, peer_tx: &mpsc::Sender<Frame>) {
        if let Ok(mut peer_senders) = self.peer_frame_txs.lock() {
            peer_senders.insert(peer_id, peer_tx.clone());
        }
    }

    fn start_agent_output_dispatcher(
        &self,
        mut receiver: mpsc::Receiver<NexoAgentOutput>,
    ) -> tokio::task::JoinHandle<Result> {
        let gateway = self.clone();
        tokio::spawn(async move {
            while let Some(message) = receiver.recv().await {
                gateway.handle_agent_output(message)?;
            }

            info!("Agent output dispatcher stopped");
            Ok(())
        })
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
/// * `frame` - Frame envelope to serialize as a text message.
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
    use nexo_core::{
        ClientInfo, DeviceInfo, GatewayProperties, NodeProperties, OperationId, UserProperties,
    };
    use nexo_ws_client::NexoConnection;
    use nexo_ws_schema::{DisconnectRequest, NexoResponse};
    use tokio::time::{Duration, timeout};

    fn gateway() -> NexoGateway {
        NexoGateway::new(GatewayProperties::default()).unwrap()
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

    #[tokio::test]
    async fn user_connect_binds_peer_and_preserves_operation_id() {
        let gateway = gateway();
        let mut state = PeerConnectionState::AwaitingConnect;
        let request = ConnectRequest::new(user_client());
        let expected_operation_id = request.operation_id;
        let frame = Frame::new(UserToGatewayMessage::Connect(request)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).await.unwrap();

        let GatewayFrameOutput::Reply(reply) = outcome else {
            panic!("expected connect reply")
        };
        assert_eq!(
            operation_id_from_user_connect(&reply),
            expected_operation_id
        );
        let PeerConnectionState::Connected { peer_id, kind } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::User);
        assert_eq!(gateway.peer(peer_id).unwrap().kind(), NexoClientKind::User);
    }

    #[tokio::test]
    async fn node_connect_binds_node_kind() {
        let gateway = gateway();
        let mut state = PeerConnectionState::AwaitingConnect;
        let request = ConnectRequest::new(node_client());
        let frame = Frame::new(NodeToGatewayMessage::Connect(request)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).await.unwrap();

        let GatewayFrameOutput::Reply(reply) = outcome else {
            panic!("expected connect reply")
        };
        let (_, message) = reply.into_parts::<GatewayToNodeMessage>().unwrap();
        assert!(matches!(
            message,
            GatewayToNodeMessage::Connect(NexoResponse::Completed { .. })
        ));
        let PeerConnectionState::Connected { peer_id, kind } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::Node);
        assert_eq!(gateway.peer(peer_id).unwrap().kind(), NexoClientKind::Node);
    }

    #[tokio::test]
    async fn first_connect_is_classified_by_nexo_client_kind() {
        let gateway = gateway();
        let mut state = PeerConnectionState::AwaitingConnect;
        let frame = Frame::new(UserToGatewayMessage::Connect(ConnectRequest::new(
            node_client(),
        )))
        .unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).await.unwrap();

        assert!(matches!(outcome, GatewayFrameOutput::Reply(_)));
        let PeerConnectionState::Connected { kind, .. } = state else {
            panic!("expected connected state")
        };
        assert_eq!(kind, NexoClientKind::Node);
    }

    #[tokio::test]
    async fn user_get_state_is_forwarded_to_agent_and_replied_from_agent_output() {
        let mut gateway = gateway();
        let agent_input_rx = gateway.agent_input_rx.take().unwrap();
        let agent_output_rx = gateway.agent_output_rx.take().unwrap();
        let _agent_task = NexoAgent::new().start(agent_input_rx, gateway.agent_output_tx.clone());
        let _dispatcher_task = gateway.start_agent_output_dispatcher(agent_output_rx);

        let user = user_client();
        let peer_id = PeerId::from_client(&user);
        let (peer_tx, mut peer_rx) = tokio::sync::mpsc::channel::<Frame>(4);
        gateway.register_peer_sender(peer_id, &peer_tx);

        gateway
            .agent_input_tx
            .try_send(NexoAgentInput::GetState {
                requester: peer_id,
                operation_id: OperationId::new(),
            })
            .unwrap();

        let frame = timeout(Duration::from_secs(1), peer_rx.recv())
            .await
            .expect("timed out waiting for get_state reply")
            .expect("peer channel closed before get_state reply");

        let (_, message) = frame.into_parts::<GatewayToUserMessage>().unwrap();
        let GatewayToUserMessage::GetState(NexoResponse::Completed { result, .. }) = message else {
            panic!("expected get_state completed response")
        };
        assert_eq!(result.user_count(), 0);
        assert_eq!(result.node_count(), 0);
    }

    #[tokio::test]
    async fn disconnect_removes_peer_and_closes_after_reply() {
        let gateway = gateway();
        let mut state = PeerConnectionState::AwaitingConnect;
        let connect_frame = Frame::new(UserToGatewayMessage::Connect(ConnectRequest::new(
            user_client(),
        )))
        .unwrap();
        gateway
            .handle_frame(&mut state, connect_frame)
            .await
            .unwrap();
        let PeerConnectionState::Connected { peer_id, .. } = state else {
            panic!("expected connected state")
        };
        assert!(gateway.peer(peer_id).is_some());
        let disconnect = DisconnectRequest::new();
        let expected_operation_id = disconnect.operation_id;
        let frame = Frame::new(UserToGatewayMessage::Disconnect(disconnect)).unwrap();

        let outcome = gateway.handle_frame(&mut state, frame).await.unwrap();

        let GatewayFrameOutput::CloseAfterReply(reply) = outcome else {
            panic!("expected close-after-reply")
        };
        let (_, message) = reply.into_parts::<GatewayToUserMessage>().unwrap();
        let GatewayToUserMessage::Disconnect(NexoResponse::Completed { operation_id, .. }) =
            message
        else {
            panic!("expected user disconnect completed response")
        };
        assert_eq!(operation_id, expected_operation_id);
        assert_eq!(state, PeerConnectionState::Disconnected);
        assert!(gateway.peer(peer_id).is_none());
    }

    #[tokio::test]
    async fn first_frame_must_be_connect() {
        let gateway = gateway();
        let mut state = PeerConnectionState::AwaitingConnect;
        let frame = Frame::new(UserToGatewayMessage::Disconnect(DisconnectRequest::new())).unwrap();

        let error = gateway.handle_frame(&mut state, frame).await.unwrap_err();

        assert!(matches!(error, Error::InvalidPeerState(_)));
        assert_eq!(state, PeerConnectionState::AwaitingConnect);
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

        let key = PeerId::new(user.client().id, user.device().id);
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
