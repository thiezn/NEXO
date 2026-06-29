use crate::ModelId;
use thiserror::Error as ThisError;

/// The result type used by `nexo-core` contracts.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Error values shared by `nexo-core` service contracts.
#[derive(Debug, Clone, PartialEq, Eq, ThisError)]
pub enum Error {
    /// The caller supplied an invalid request or malformed payload.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// The human-readable validation error message.
        message: String,
    },

    /// The requested operation is not supported by the selected implementation.
    #[error("unsupported feature: {feature}")]
    UnsupportedFeature {
        /// The unsupported feature name.
        feature: String,
    },

    /// The target service is in a state that prevents the requested operation.
    #[error("invalid state: {message}")]
    InvalidState {
        /// The human-readable state violation message.
        message: String,
    },

    /// An error occured during inference
    #[error("inference error: {message}")]
    Inference {
        /// The human-readable inference error message.
        message: String,
    },

    /// The requested model was not found in the available model definitions.
    #[error("model not found: {model_id}")]
    ModelNotFound {
        /// The model ID that was not found.
        model_id: ModelId,
    },

    /// The requested tool was not found in the available tool definitions.
    #[error("tool not found: {name}")]
    ToolNotFound {
        /// The tool name that was not found.
        name: String,
    },
}
