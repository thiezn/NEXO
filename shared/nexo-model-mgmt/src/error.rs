//! Error types for model management commands and helpers.

use std::error::Error as StdError;
use std::fmt;

/// Result alias for model management operations.
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Errors produced by model management operations.
#[derive(Debug)]
pub enum Error {
    /// The shared CLI runtime returned an error.
    Cli(cli_helpers::Error),
    /// A model download failed.
    Download(crate::pull::DownloadError),
    /// A requested model is not known by the local manifest registry.
    UnknownModel {
        /// The unknown model or category requested by the user.
        model: String,
        /// Known model names that can be requested.
        known: Vec<String>,
    },
    /// An I/O operation failed.
    Io(std::io::Error),
    /// A tokio runtime could not be created for a synchronous command entry point.
    Runtime(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(source) => write!(f, "models command failed: {source}"),
            Self::Download(source) => write!(
                f,
                "model download failed: {source}. Do you have hf_token.txt in .nexo?"
            ),
            Self::UnknownModel { model, known } => write!(
                f,
                "unknown model or category '{model}'. Known models: {}",
                known.join(", ")
            ),
            Self::Io(source) => write!(f, "I/O error: {source}"),
            Self::Runtime(source) => write!(f, "failed to create command runtime: {source}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Cli(source) => Some(source),
            Self::Download(source) => Some(source),
            Self::Io(source) | Self::Runtime(source) => Some(source),
            Self::UnknownModel { .. } => None,
        }
    }
}

impl From<cli_helpers::Error> for Error {
    fn from(error: cli_helpers::Error) -> Self {
        Self::Cli(error)
    }
}

impl From<crate::pull::DownloadError> for Error {
    fn from(error: crate::pull::DownloadError) -> Self {
        Self::Download(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
