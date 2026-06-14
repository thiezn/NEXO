use crate::ModelSelection;
use crate::ids::ModelId;
use crate::model::{ModelDefinition, ModelRuntimeState};

/// A catalog of models available to higher-level Nexo services.
pub trait ModelRegistry: Send + Sync {
    /// Returns all known model descriptors.
    fn list_models(&self) -> Vec<ModelDefinition>;

    /// Returns the descriptor for a specific model identifier, if it exists.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier to resolve.
    fn get_model(&self, model_id: &ModelId) -> Option<ModelDefinition>;

    /// Resolves a model selection request to a concrete model descriptor.
    ///
    /// # Arguments
    ///
    /// * `selection` - The model selection criteria to resolve.
    fn resolve_model(&self, selection: &ModelSelection) -> Option<ModelDefinition>;

    /// Returns the current runtime state for a model, if it is known.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier to inspect.
    fn model_state(&self, model_id: &ModelId) -> Option<ModelRuntimeState>;
}
