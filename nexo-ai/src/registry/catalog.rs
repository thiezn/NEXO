use std::collections::BTreeMap;

use nexo_core::{ModelCapability, ModelDescriptor, ModelId, ModelSelection};

use crate::{Error, ModelRuntimeState, Result};

/// An immutable in-memory model registry backed by configured descriptors.
#[derive(Debug, Clone)]
pub struct StaticModelRegistry {
    descriptors: BTreeMap<ModelId, ModelDescriptor>,
}

impl StaticModelRegistry {
    /// Creates a new registry from the configured descriptors.
    ///
    /// # Arguments
    ///
    /// * `descriptors` - The model descriptors to expose through the registry.
    pub fn new(descriptors: Vec<ModelDescriptor>) -> Result<Self> {
        if descriptors.is_empty() {
            return Err(Error::EmptyModelCatalog);
        }

        let mut by_id = BTreeMap::new();
        for descriptor in descriptors {
            let model_id = descriptor.id.clone();
            if by_id.insert(model_id.clone(), descriptor).is_some() {
                return Err(Error::DuplicateModelId { model_id });
            }
        }

        Ok(Self { descriptors: by_id })
    }

    /// Returns all configured descriptors in deterministic identifier order.
    pub fn list_models(&self) -> Vec<ModelDescriptor> {
        self.descriptors.values().cloned().collect()
    }

    /// Returns the configured descriptor for the given model, if it exists.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier to look up.
    pub fn get_model(&self, model_id: &ModelId) -> Option<ModelDescriptor> {
        self.descriptors.get(model_id).cloned()
    }

    /// Resolves a `nexo-core` model selection request to a concrete descriptor.
    ///
    /// # Arguments
    ///
    /// * `selection` - The selection criteria to resolve.
    pub fn resolve_model(&self, selection: &ModelSelection) -> Option<ModelDescriptor> {
        if let Some(model_id) = &selection.specific_model {
            return self.get_model(model_id).filter(|descriptor| {
                supports_all(&descriptor.capabilities, &selection.required_capabilities)
            });
        }

        self.descriptors
            .values()
            .filter(|descriptor| {
                supports_all(&descriptor.capabilities, &selection.required_capabilities)
            })
            .max_by_key(|descriptor| {
                selection
                    .preferred_capabilities
                    .iter()
                    .filter(|capability| descriptor.capabilities.contains(capability))
                    .count()
            })
            .cloned()
    }

    /// Returns the static runtime state used before a dynamic runtime is attached.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier to inspect.
    pub fn model_state(&self, model_id: &ModelId) -> Option<ModelRuntimeState> {
        self.descriptors
            .contains_key(model_id)
            .then_some(ModelRuntimeState::Unloaded)
    }
}

fn supports_all(available: &[ModelCapability], required: &[ModelCapability]) -> bool {
    required
        .iter()
        .all(|capability| available.contains(capability))
}

#[cfg(test)]
mod tests {
    use nexo_core::{MetadataMap, ModelModalities, RoleStrategy, SupportedModality};

    use super::*;

    #[test]
    fn resolves_specific_model_with_required_capability() {
        let descriptor = descriptor("chat", vec![ModelCapability::TextGeneration]);
        let registry = StaticModelRegistry::new(vec![descriptor.clone()]).unwrap();

        let resolved = registry.resolve_model(&ModelSelection {
            specific_model: Some(ModelId::from("chat")),
            required_capabilities: vec![ModelCapability::TextGeneration],
            preferred_capabilities: Vec::new(),
            runtime_preference: Default::default(),
        });

        assert_eq!(resolved, Some(descriptor));
    }

    #[test]
    fn prefers_model_with_more_preferred_capabilities() {
        let text = descriptor("text", vec![ModelCapability::TextGeneration]);
        let tool = descriptor(
            "tool",
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
            ],
        );
        let registry = StaticModelRegistry::new(vec![text, tool.clone()]).unwrap();

        let resolved = registry.resolve_model(&ModelSelection {
            specific_model: None,
            required_capabilities: vec![ModelCapability::TextGeneration],
            preferred_capabilities: vec![ModelCapability::ToolCalling],
            runtime_preference: Default::default(),
        });

        assert_eq!(resolved, Some(tool));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let first = descriptor("chat", vec![ModelCapability::TextGeneration]);
        let second = descriptor("chat", vec![ModelCapability::Embeddings]);

        assert!(matches!(
            StaticModelRegistry::new(vec![first, second]),
            Err(Error::DuplicateModelId { .. })
        ));
    }

    fn descriptor(id: &str, capabilities: Vec<ModelCapability>) -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId::from(id),
            display_name: id.to_string(),
            provider: Some("test".to_string()),
            capabilities,
            modalities: ModelModalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            role_strategy: RoleStrategy::Default,
            context_window_tokens: Some(4096),
            max_output_tokens: Some(1024),
            metadata: MetadataMap::new(),
        }
    }
}
