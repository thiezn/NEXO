use hf_hub::api::tokio::ApiError;
use nexo_core::{ModelId, ModelRuntimeState};
use std::path::PathBuf;
use thiserror::Error as ThisError;

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

    /// The requested model is known but is not currently loaded into the runtime.
    #[error("model `{model_id}` is not loaded")]
    ModelNotLoaded {
        /// The unloaded model identifier.
        model_id: ModelId,

        /// The current runtime state of the model
        current_state: ModelRuntimeState,
    },

    /// The requested model is known but could not be unloaded
    #[error("model `{model_id}` is not unloaded")]
    ModelNotUnloaded {
        /// The unloaded model identifier.
        model_id: ModelId,

        /// The current runtime state of the model
        current_state: ModelRuntimeState,
    },

    // The requested model is already running an inference request and cannot accept concurrent requests.
    #[error("model `{model_id}` is already running an inference request")]
    ModelBusy {
        /// The busy model identifier.
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

    /// The current model runtime is not implemented for the requested model.
    #[error("runtime not implemented for '{model_id}'")]
    RuntimeNotImplemented { model_id: ModelId },

    /// A runtime interaction failed before a request stream was accepted.
    #[error("runtime error: {message}")]
    Runtime {
        /// The human-readable runtime failure message.
        message: String,
    },

    /// The source repository is gated and requires approval.
    #[error("model requires access approval on Hugging Face: {repo}")]
    GatedModel {
        /// The gated Hugging Face repository.
        repo: String,
    },

    /// The source repository requires authentication.
    #[error("authentication required for Hugging Face repository {repo}")]
    Unauthorized {
        /// The Hugging Face repository that rejected unauthenticated access.
        repo: String,
    },

    /// A file failed to download.
    #[error("download failed for {filename} from {repo}: {source}")]
    DownloadFailed {
        /// The Hugging Face repository containing the file.
        repo: String,
        /// The repository-relative filename that failed to download.
        filename: String,
        /// The underlying Hugging Face API error.
        source: ApiError,
    },

    /// A pattern selector did not match any remote files.
    #[error("no files in {repo} matched selector {selector}")]
    NoFilesMatched {
        /// The Hugging Face repository being inspected.
        repo: String,
        /// The selector that matched no files.
        selector: String,
    },

    /// Placing a downloaded file into the clean local model directory failed.
    #[error("failed to place downloaded file: {0}")]
    FilePlacement(String),

    /// SHA-256 verification could not read the file.
    #[error("failed to verify SHA-256 for {path}: {source}")]
    VerifyHash {
        /// The local file path being verified.
        path: PathBuf,
        /// The underlying file read error.
        source: std::io::Error,
    },

    /// Hugging Face API client setup failed.
    #[error("failed in Hugging Face API client: {0}")]
    Hf(#[from] ApiError),

    /// A standard I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// A `nexo-core` contract rejected the operation.
    #[error(transparent)]
    Core(#[from] nexo_core::Error),

    /// A 'anytts' runtime error occurred.
    #[error(transparent)]
    AnyTts(#[from] any_tts::TtsError),
}
