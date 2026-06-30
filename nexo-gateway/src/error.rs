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
}
