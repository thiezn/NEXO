use thiserror::Error as ThisError;

use nexo_core::ModelId;

/// The result type used by the `nexo-ai` crate.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Error values returned while configuring or building `nexo-ai` services.
#[derive(Debug, ThisError)]
pub enum Error {
    /// The provided model list was empty.
    #[error("at least one model must be configured")]
    EmptyModelCatalog,

    /// A duplicate model identifier was found in configuration.
    #[error("duplicate model identifier `{model_id}`")]
    DuplicateModelId {
        /// The conflicting model identifier.
        model_id: ModelId,
    },

    /// The configured model selection did not resolve to any descriptor.
    #[error("could not resolve model selection: {message}")]
    UnresolvedModelSelection {
        /// The human-readable selection failure message.
        message: String,
    },

    /// The selected model could not be found in the configured catalog.
    #[error("unknown model `{model_id}`")]
    UnknownModel {
        /// The missing model identifier.
        model_id: ModelId,
    },

    /// The requested feature is not implemented for the current runtime setup.
    #[error("unsupported feature: {feature}")]
    UnsupportedFeature {
        /// The human-readable feature description.
        feature: String,
    },

    /// The request type is not supported by this crate yet.
    #[error("unsupported request type: {kind}")]
    UnsupportedRequest {
        /// The unsupported request kind.
        kind: &'static str,
    },

    /// The request contains message content that the adapter cannot currently map.
    #[error("unsupported message part: {part}")]
    UnsupportedMessagePart {
        /// The unsupported message part description.
        part: &'static str,
    },

    /// The request contains an invalid tool definition or tool call payload.
    #[error("invalid tool payload for `{tool_name}`: {message}")]
    InvalidToolPayload {
        /// The tool name associated with the invalid payload.
        tool_name: String,

        /// The validation failure message.
        message: String,
    },

    /// A configuration load or save operation failed through `cli-helpers`.
    #[error("configuration error: {message}")]
    Config {
        /// The human-readable configuration failure message.
        message: String,
    },

    /// A `mistralrs-core` interaction failed before a request stream was accepted.
    #[error("mistral runtime error: {message}")]
    MistralRuntime {
        /// The human-readable runtime failure message.
        message: String,
    },

    /// A standard I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// A `nexo-core` contract rejected the operation.
    #[error(transparent)]
    Core(#[from] nexo_core::Error),
}

impl Error {
    /// Converts the crate-local error into a `nexo-core` contract error.
    pub(crate) fn into_core_error(self) -> nexo_core::Error {
        match self {
            Self::Core(error) => error,
            Self::UnsupportedFeature { feature } => {
                nexo_core::Error::UnsupportedFeature { feature }
            }
            Self::UnsupportedRequest { kind } => nexo_core::Error::UnsupportedFeature {
                feature: kind.to_string(),
            },
            Self::UnknownModel { model_id } => nexo_core::Error::InvalidRequest {
                message: format!("unknown model `{model_id}`"),
            },
            Self::UnresolvedModelSelection { message }
            | Self::Config { message }
            | Self::MistralRuntime { message } => nexo_core::Error::InvalidState { message },
            Self::UnsupportedMessagePart { part } => nexo_core::Error::InvalidState {
                message: part.to_string(),
            },
            Self::InvalidToolPayload { tool_name, message } => nexo_core::Error::InvalidRequest {
                message: format!("invalid tool payload for `{tool_name}`: {message}"),
            },
            Self::EmptyModelCatalog => nexo_core::Error::InvalidState {
                message: "at least one model must be configured".to_string(),
            },
            Self::DuplicateModelId { model_id } => nexo_core::Error::InvalidState {
                message: format!("duplicate model identifier `{model_id}`"),
            },
            Self::Io(error) => nexo_core::Error::InvalidState {
                message: error.to_string(),
            },
            Self::Json(error) => nexo_core::Error::InvalidRequest {
                message: error.to_string(),
            },
        }
    }
}
