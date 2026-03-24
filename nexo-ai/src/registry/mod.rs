pub mod manifest;
pub mod models;

pub use manifest::{
    AiComponent, AiModelManifest, find_manifest, known_manifests, manifests_for_category,
};
pub use models::{list_models, ModelEntry};
