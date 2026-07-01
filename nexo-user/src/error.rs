//! Error types for model management commands and helpers.
use nexo_ws_schema::UserToGatewayMessage;
use thiserror::Error as ThisError;

/// Result alias for model management operations.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Errors produced by model management operations.
#[derive(Debug, ThisError)]
pub enum Error {
    /// An unexpected error occurred.
    /// Should not be used...!
    #[error("Unexpected error: {0}")]
    Other(String),

    /// An I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An error occured in the WebSocket Client
    #[error(transparent)]
    WsClient(#[from] nexo_ws_client::Error),

    /// An error occured in the CLI helpers library
    #[error(transparent)]
    CliHelpers(#[from] cli_helpers::Error),

    /// An error occured in the ws-schema library
    #[error(transparent)]
    WsSchema(#[from] nexo_ws_schema::Error),

    /// Error occures in nexo_core
    #[error(transparent)]
    NexoCore(#[from] nexo_core::Error),

    /// An error occured in the tokio library
    #[error(transparent)]
    SendError(#[from] tokio::sync::mpsc::error::SendError<UserToGatewayMessage>),
}
