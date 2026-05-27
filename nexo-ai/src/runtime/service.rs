use nexo_core::{InferenceEngine, InferenceRequest, InferenceStream, ModelId, ModelRegistry};

use crate::runtime::engine::MistralRuntime;
use crate::{Error, NexoAiConfig, StaticModelRegistry};

/// The library-first `nexo-ai` service surface.
#[derive(Debug, Clone)]
pub struct NexoAi {
    registry: StaticModelRegistry,
    runtime: MistralRuntime,
}

impl NexoAi {
    /// Creates a new builder for the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The declarative runtime configuration.
    pub fn builder(config: NexoAiConfig) -> crate::runtime::NexoAiBuilder {
        crate::runtime::NexoAiBuilder::new(config)
    }

    /// Builds a service from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The declarative runtime configuration.
    pub async fn from_config(config: NexoAiConfig) -> crate::Result<Self> {
        let descriptors = config
            .models
            .iter()
            .map(|model| model.descriptor.clone())
            .collect();
        let registry = StaticModelRegistry::new(descriptors)?;
        let runtime = MistralRuntime::from_config(&config).await?;
        Ok(Self { registry, runtime })
    }

    fn resolve_request_model(
        &self,
        request: &InferenceRequest,
    ) -> crate::Result<nexo_core::ModelDescriptor> {
        let selection = match request {
            InferenceRequest::Generate(request) => &request.model,
            InferenceRequest::Embed(request) => &request.model,
            InferenceRequest::GenerateImage(request) => &request.model,
            InferenceRequest::GenerateSpeech(request) => &request.model,
            InferenceRequest::Tokenize(request) => &request.model,
            InferenceRequest::Detokenize(request) => &request.model,
        };

        if let Some(descriptor) = self.registry.resolve_model(selection) {
            return Ok(descriptor);
        }

        if let Some(model_id) = &selection.specific_model {
            return Err(Error::UnknownModel {
                model_id: model_id.clone(),
            });
        }

        Err(Error::UnresolvedModelSelection {
            message: "no configured model satisfies the requested selection".to_string(),
        })
    }

    fn model_state_from_runtime(&self, model_id: &ModelId) -> Option<nexo_core::ModelRuntimeState> {
        self.runtime.model_state(model_id)
    }

    /// Returns the immutable model registry view used by this service.
    pub fn registry(&self) -> &StaticModelRegistry {
        &self.registry
    }

    /// Returns the static registry entry for a specific model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier to look up.
    pub fn model(&self, model_id: &ModelId) -> Option<nexo_core::ModelDescriptor> {
        self.registry.get_model(model_id)
    }
}

impl ModelRegistry for NexoAi {
    fn list_models(&self) -> Vec<nexo_core::ModelDescriptor> {
        self.registry.list_models()
    }

    fn get_model(&self, model_id: &ModelId) -> Option<nexo_core::ModelDescriptor> {
        self.registry.get_model(model_id)
    }

    fn resolve_model(
        &self,
        selection: &nexo_core::ModelSelection,
    ) -> Option<nexo_core::ModelDescriptor> {
        self.registry.resolve_model(selection)
    }

    fn model_state(&self, model_id: &ModelId) -> Option<nexo_core::ModelRuntimeState> {
        self.model_state_from_runtime(model_id)
            .or_else(|| self.registry.model_state(model_id))
    }
}

impl InferenceEngine for NexoAi {
    fn submit(&self, request: InferenceRequest) -> nexo_core::Result<InferenceStream> {
        let descriptor = self
            .resolve_request_model(&request)
            .map_err(Error::into_core_error)?;
        self.runtime
            .submit(descriptor, request)
            .map_err(Error::into_core_error)
    }
}
