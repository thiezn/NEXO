use thiserror::Error;

/// Errors returned by the websocket client helpers.
#[derive(Debug, Error)]
pub enum ClientError {
    /// Underlying websocket transport failure.
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    /// Gateway protocol validation or negotiation failure.
    #[error("Protocol error: {0}")]
    Protocol(#[from] nexo_ws_schema::WsError),

    /// JSON serialization or deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The websocket closed before the expected frame arrived.
    #[error("Connection closed")]
    Closed,

    /// The gateway handshake failed.
    #[error("Handshake failed: {0}")]
    Handshake(String),

    /// A client operation exceeded its timeout.
    #[error("Timeout")]
    Timeout,
}

/// Result alias for websocket client operations.
pub type Result<T = ()> = std::result::Result<T, ClientError>;
