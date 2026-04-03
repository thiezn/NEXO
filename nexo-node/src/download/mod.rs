pub mod paths;

// Re-export nexo-ai's download infrastructure.
pub use nexo_ai::download::manifest::{Component, storage_path};
pub use nexo_ai::download::paths::default_models_dir;
pub use nexo_ai::download::pull::pull_model;
pub use nexo_ai::registry::{find_manifest, known_manifests};
