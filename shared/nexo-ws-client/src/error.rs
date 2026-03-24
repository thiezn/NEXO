use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("Protocol error: {0}")]
    Protocol(#[from] nexo_ws_schema::WsError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Connection closed")]
    Closed,

    #[error("Handshake failed: {0}")]
    Handshake(String),

    #[error("Timeout")]
    Timeout,
}

pub type Result<T = ()> = std::result::Result<T, ClientError>;
