//! Local model catalog, download, and CLI command support for NEXO.

pub mod command;
pub mod error;
pub mod manifest;
pub mod paths;
pub mod pull;
pub mod registry;

pub use command::{ModelsAction, ModelsCommand};
pub use error::{Error, Result};
pub use manifest::{
    ModelComponent, ModelFile, ModelFileSelector, ModelManifest, sanitize_model_name, storage_path,
};
pub use paths::{default_models_dir, model_storage_dir, resolve_model_storage_dir};
pub use pull::{DownloadError, pull_model, verify_sha256};
pub use registry::{
    ModelEntry, capability_label, find_manifest, known_manifests, manifests_for_capability,
    manifests_for_modality,
};
