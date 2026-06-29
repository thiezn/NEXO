//! Error types for model management commands and helpers.
use nexo_core::OperationId;
use nexo_ws_schema::NodeToGatewayMessage;
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

    /// A requested model is not known by the local manifest registry.
    #[error("unknown model or category '{model}'")]
    UnknownModel {
        /// The unknown model or category requested by the user.
        model: String,
        /// Known model names that can be requested.
        known: Vec<String>,
    },

    /// An error occurred during tool registration.
    #[error("Tool registration error")]
    ToolRegistration {
        /// The operation ID associated with the tool registration request.
        operation_id: OperationId,

        /// The human-readable error message describing the tool registration failure.
        error: String,
    },

    /// An I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An NexoAI operation failed
    #[error(transparent)]
    NexoAI(#[from] nexo_ai::Error),

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
    SendError(#[from] tokio::sync::mpsc::error::SendError<NodeToGatewayMessage>),
}
