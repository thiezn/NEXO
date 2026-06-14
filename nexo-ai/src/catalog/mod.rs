//! The Catalog manages the download and storage of model artifacts

pub mod definitions;
pub mod model_catalog;
pub mod model_file;
pub mod model_manifest;

pub use definitions::ALL_MANIFESTS;
pub use model_catalog::ModelCatalog;
pub use model_file::ModelFile;
pub use model_manifest::ModelManifest;
