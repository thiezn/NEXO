//! The Catalog manages the download and storage of model artifacts
//!
//! The specific manifests are statically defined in the manifests sub module. ModelDefinitions
//! are defined in nexo-core as those properties are used across the whole workspace.

pub(crate) mod download;
pub mod manifests;
pub mod model_catalog;
pub mod model_file;
pub mod model_manifest;
pub(crate) mod paths;
pub(crate) mod progress;

pub use download::DownloadOptions;
pub use manifests::ALL_MANIFESTS;
pub use model_catalog::ModelCatalog;
pub use model_file::{ModelFile, ModelFileKind};
pub use model_manifest::ModelManifest;
pub use progress::{CatalogDownloadProgress, FileDownloadProgress, NoopDownloadProgress};
