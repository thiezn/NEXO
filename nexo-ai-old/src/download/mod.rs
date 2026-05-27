pub mod manifest;
pub mod paths;

#[cfg(feature = "download")]
pub mod pull;

// Re-exports for convenience.
pub use manifest::{Component, ModelFile, ModelManifest};
pub use paths::default_models_dir;

#[cfg(feature = "download")]
pub use pull::{DownloadError, pull_model, verify_sha256};
