use super::model_file::ModelFile;
use super::paths::{default_models_dir, is_relative_storage_path};
use crate::Result;
use nexo_core::{ModelDefinition, ModelFamily, ModelId};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// A single logical AI Model used for inference
#[derive(Clone)]
pub struct ModelManifest {
    /// The definition of the model, including its unique identifier and metadata.
    definition: ModelDefinition,

    /// The estimated size of the model when loaded in memory, in gigabytes.
    ///
    /// This excludes KV Cache memory, which can be significant for large models
    /// but is not required to be allocated upfront.
    ram_size_gb: f32,

    /// The files associated with this model, such as weights, configuration, and tokenizer files.
    files: Vec<ModelFile>,

    /// The storage folder relative to the base storage path for local model artifacts.
    ///
    /// This can be overwritten to allow multiple models to share the same underlying files,
    /// for example when different models are just different configurations of the same
    /// underlying weights.
    ///
    /// If `None`, the storage folder will default to the model ID, which is a common case
    /// for most models.
    storage_folder: Option<String>,
}

/// Shared local storage key used for downloaded artifacts.
impl ModelManifest {
    /// Creates a new ModelManifest with the given Model Id, RAM size, and associated files.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The unique identifier for the model.
    /// * `ram_size_gb` - The estimated RAM size in gigabytes required to load this model.
    /// * `files` - The list of files associated with this model manifest.
    pub fn new(model_id: ModelId, ram_size_gb: f32, files: Vec<ModelFile>) -> Self {
        Self {
            definition: ModelDefinition::new(model_id),
            ram_size_gb,
            files,
            storage_folder: None,
        }
    }

    /// Creates a new ModelManifest with the given Model Id, RAM size, associated files
    /// and override storage folder.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The unique identifier for the model.
    /// * `ram_size_gb` - The estimated RAM size in gigabytes required to load this model.
    /// * `files` - The list of files associated with this model manifest.
    /// * `storage_folder` - The storage folder name relative to the base storage path.
    pub fn new_with_folder(
        model_id: ModelId,
        ram_size_gb: f32,
        files: Vec<ModelFile>,
        storage_folder: impl Into<String>,
    ) -> Self {
        Self {
            definition: ModelDefinition::new(model_id),
            ram_size_gb,
            files,
            storage_folder: Some(storage_folder.into()),
        }
    }

    /// Returns the storage folder for this model manifest, which is either the explicitly
    /// set storage folder or defaults to the model ID.
    pub fn storage_folder(&self) -> String {
        self.storage_folder
            .clone()
            .unwrap_or_else(|| self.definition.id().clone().to_string())
    }

    /// Returns whether this model manifest has been downloaded and verified locally.
    ///
    /// File verification runs in parallel across a Rayon thread-pool to speed up
    /// SHA-256 validation for manifests with many or large files.
    pub fn is_downloaded(&self) -> bool {
        let Ok(model_dir) = self.model_dir() else {
            return false;
        };

        self.files
            .par_iter()
            .all(|file| file.is_downloaded(&model_dir))
    }

    /// Returns whether this model manifest appears present locally using fast metadata checks.
    ///
    /// This does not compute SHA-256 for every file and is intended for quick status views.
    pub fn is_present_locally(&self) -> bool {
        let Ok(model_dir) = self.model_dir() else {
            return false;
        };

        self.files.iter().all(|file| file.is_present(&model_dir))
    }

    /// Returns the unique identifier of the model defined by this manifest.
    pub fn model_id(&self) -> &ModelId {
        &self.definition.id()
    }

    /// Returns the model definition associated with this manifest.
    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    /// Returns the model family of the associated Model.
    pub fn family(&self) -> ModelFamily {
        self.model_id().family()
    }

    /// Returns the capabilities of the associated Model.
    pub fn capabilities(&self) -> &[nexo_core::ModelCapability] {
        self.definition.capabilities()
    }

    /// Returns the estimated RAM size in gigabytes required to load this model.
    ///
    /// This excludes KV Cache memory, which can be significant for large models
    /// but is not required to be allocated upfront.
    pub fn ram_size_gb(&self) -> f32 {
        self.ram_size_gb
    }

    /// Returns the total download size in bytes for all files associated with this model.
    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|file| file.size_bytes()).sum()
    }

    /// Returns the list of files associated with this model manifest.
    pub fn files(&self) -> &[ModelFile] {
        &self.files
    }

    /// Resolves the local storage directory for this model manifest.
    pub(crate) fn model_dir(&self) -> Result<PathBuf> {
        let storage_folder = self.storage_folder();
        let storage_path = Path::new(&storage_folder);
        if !is_relative_storage_path(storage_path) {
            return Err(crate::Error::InvalidStoragePath {
                path: storage_folder,
            });
        }

        Ok(default_models_dir().join(storage_path))
    }
}
