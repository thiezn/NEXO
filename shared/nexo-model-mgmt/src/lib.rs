//! Local model catalog, download, and CLI command support for NEXO.

pub mod error;
pub mod manifest;
pub mod paths;
pub mod pull;
pub mod registry;

pub use error::{Error, Result};
pub use manifest::{
    AnyTtsManifestBinding, AnyTtsManifestEngine, ManifestModelDataType, ManifestRuntimeBinding,
    MistralRsAutoManifestLoader, MistralRsDiffusionManifestLoader, MistralRsGgufManifestLoader,
    MistralRsManifestBinding, MistralRsManifestLoader, MistralRsSpeechManifestLoader,
    ModelComponent, ModelFile, ModelFileSelector, ModelManifest, MoldManifestBinding,
    MoldManifestLoader, sanitize_model_name, storage_path, storage_path_for_file,
};
pub use paths::{default_models_dir, model_storage_dir, resolve_model_storage_dir};
pub use pull::{DownloadError, pull_model, verify_sha256};
pub use registry::{
    ModelEntry, capability_label, find_manifest, known_manifests, manifests_for_capability,
};
