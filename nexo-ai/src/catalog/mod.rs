//! The Catalog manages the download and storage of model artifacts
//!
//! The specific manifests are statically defined in the manifests sub module. ModelDefinitions
//! are defined in nexo-core as those properties are used across the whole workspace.

/// The Catalog is responsible for downloading and storing model artifacts, as well as providing access to the model definitions. It uses the manifests defined in the manifests submodule to determine which models are available for download and how to download them.
pub(crate) mod download;

/// The manifests submodule contains the definitions of the available model manifests. Each manifest defines a set of models that can be downloaded and their associated metadata.
pub mod manifests;

/// The model_catalog submodule provides the ModelCatalog struct, which is responsible for managing the collection of available models and their associated metadata.
pub mod model_catalog;

/// The model_file submodule defines the ModelFile struct and the ModelFileKind enum, which represent the individual model files that can be downloaded and their types.
pub mod model_file;

/// The model_manifest submodule defines the ModelManifest struct, which represents the metadata associated with a specific model manifest, including its name, version, and the list of available models.
pub mod model_manifest;

/// The paths submodule provides utilities for managing file paths related to model artifacts, including determining where to store downloaded models and how to locate them on the filesystem.
pub(crate) mod paths;

/// The progress submodule defines the CatalogDownloadProgress, FileDownloadProgress, and NoopDownloadProgress structs, which are used to track the progress of model downloads and provide feedback to the user.
pub(crate) mod progress;

pub use download::DownloadOptions;
pub use manifests::ALL_MANIFESTS;
pub use model_catalog::ModelCatalog;
pub use model_file::{ModelFile, ModelFileKind};
pub use model_manifest::ModelManifest;
pub use progress::{CatalogDownloadProgress, FileDownloadProgress, NoopDownloadProgress};
