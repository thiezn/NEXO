//! Model registry metadata and model availability helpers.

mod availability;
pub mod manifest;
pub mod models;

#[cfg(feature = "download")]
pub use availability::detect_available_models;
pub use manifest::{
    AiComponent, AiModelManifest, CandleBackend, ModelFamily, ModelRuntime, OpenAiProvider,
    find_manifest, known_manifests, manifests_for_category,
};
pub use models::{ModelEntry, list_models};
