use thiserror::Error as ThisError;

/// Result alias for model management operations.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Errors produced by model management operations.
#[derive(Debug, ThisError)]
pub enum Error {
    /// An error occured in the CLI helpers library
    #[error(transparent)]
    CliHelpers(#[from] cli_helpers::Error),

    /// An error occured in the ws-schema library
    #[error(transparent)]
    WsSchema(#[from] nexo_ws_schema::Error),

    /// Error occures in nexo_core
    #[error(transparent)]
    NexoCore(#[from] nexo_core::Error),

    /// Error occured while accepting or handling a TCP connection.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Error occured while handling a WebSocket connection.
    #[error(transparent)]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    /// Error occured while parsing or serializing JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Error occurred while queueing input for the NexoAgent.
    #[error(transparent)]
    AgentInputChannel(#[from] tokio::sync::mpsc::error::TrySendError<crate::agent::NexoAgentInput>),

    /// Error occurred while forwarding output from the NexoAgent.
    #[error(transparent)]
    AgentOutputChannel(#[from] tokio::sync::mpsc::error::SendError<crate::agent::NexoAgentOutput>),

    /// Error occurred while waiting for a spawned task to complete.
    #[error(transparent)]
    TaskJoin(#[from] tokio::task::JoinError),

    /// Error occurred while queueing a directed frame for a connected peer.
    #[error(transparent)]
    PeerFrameChannel(#[from] tokio::sync::mpsc::error::TrySendError<nexo_ws_schema::Frame>),

    /// Received a frame that is invalid for the current peer connection state.
    #[error("invalid peer state: {0}")]
    InvalidPeerState(String),
}
