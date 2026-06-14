use super::ModelManifest;
use super::definitions::ALL_MANIFESTS;
use crate::Result;
use nexo_core::ModelId;
use std::collections::BTreeMap;

/// The ModelCatalog manages the download and storage of model artifacts.
pub(crate) struct ModelCatalog {
    /// Any supported model manifest keyed by model ID.
    all_manifests: BTreeMap<ModelId, ModelManifest>,
}

impl ModelCatalog {
    /// Creates a new ModelCatalog with the static full list of ModelManifests.
    pub fn new() -> Self {
        Self {
            all_manifests: ALL_MANIFESTS
                .iter()
                .map(|m| (m.model_id().clone(), m.clone()))
                .collect(),
        }
    }

    /// Retrieves a model manifest by its ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to retrieve the manifest for.
    pub fn get_model_manifest(&self, model_id: &ModelId) -> Option<&ModelManifest> {
        self.all_manifests.get(model_id)
    }

    /// Lists all downloaded model manifests.
    pub fn list_downloaded_manifests(&self) -> Vec<&ModelManifest> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(manifest))
            .collect()
    }

    /// Retrieves a model definition by its ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to retrieve the definition for.
    pub fn get_model_definition(&self, model_id: &ModelId) -> Option<&ModelDefinition> {
        self.get_model_manifest(model_id)
            .map(|manifest| &manifest.definition())
    }

    /// Lists all downloaded model definitions.
    pub fn list_available_models(&self) -> Vec<&ModelId> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(manifest.model_id()))
            .collect()
    }

    /// Lists all downloaded model definitions.
    pub fn list_available_model_definitions(&self) -> Vec<&ModelDefinition> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(&manifest.definition()))
            .collect()
    }

    /// Downloads a model manifest by its ID, returning an error if the model
    /// is not found or fails to download.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to download.
    pub async fn download_model(&self, model_id: &ModelId) -> Result {
        let manifest =
            self.get_model_manifest(model_id)
                .ok_or_else(|| crate::Error::ModelNotFound {
                    model_id: model_id.clone(),
                })?;

        if manifest.is_downloaded() {
            return Ok(());
        }

        manifest.download().await?;
        Ok(())
    }
}
