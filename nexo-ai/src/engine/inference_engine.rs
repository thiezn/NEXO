use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use nexo_core::{InferenceRequest, InferenceRuntime, InferenceStream, ModelId, ModelRegistry};
use tokio::sync::Mutex;

use crate::engine::runtime_manager::RuntimeManager;
use crate::{Error, InferenceEngineConfig, StaticModelRegistry};

/// The library-first `nexo-ai` service surface.
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    registry: StaticModelRegistry,
    runtime_manager: Arc<Mutex<RuntimeManager>>,
    loaded_model_ids: Arc<RwLock<BTreeSet<ModelId>>>,
}

impl InferenceEngine {
    /// Creates an inference engine from declarative configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The declarative runtime configuration.
    pub async fn new(config: InferenceEngineConfig) -> crate::Result<Self> {
        let descriptors = config
            .models
            .iter()
            .map(|model| model.descriptor.clone())
            .collect();
        let registry = StaticModelRegistry::new(descriptors)?;
        let runtime_manager = RuntimeManager::new(&config);
        Ok(Self {
            registry,
            runtime_manager: Arc::new(Mutex::new(runtime_manager)),
            loaded_model_ids: Arc::new(RwLock::new(BTreeSet::new())),
        })
    }

    fn model_state_from_runtime(&self, model_id: &ModelId) -> Option<nexo_core::ModelRuntimeState> {
        self.loaded_model_ids.read().ok().and_then(|loaded| {
            loaded
                .contains(model_id)
                .then_some(nexo_core::ModelRuntimeState::Loaded)
        })
    }

    fn set_loaded_snapshot(&self, model_ids: BTreeSet<ModelId>) {
        if let Ok(mut loaded) = self.loaded_model_ids.write() {
            *loaded = model_ids;
        }
    }

    /// Loads a configured model into memory without exposing backend details to callers.
    pub async fn load_model(
        &self,
        model_id: &ModelId,
        runtime_kind: InferenceRuntime,
    ) -> crate::Result<()> {
        let mut runtime_manager = self.runtime_manager.lock().await;
        runtime_manager.load_model(model_id, runtime_kind).await?;
        self.set_loaded_snapshot(runtime_manager.loaded_model_ids());
        Ok(())
    }

    /// Unloads a configured model and returns whether it was previously loaded.
    pub async fn unload_model(&self, model_id: &ModelId) -> crate::Result<bool> {
        let mut runtime_manager = self.runtime_manager.lock().await;
        let unloaded = runtime_manager.unload_model(model_id)?;
        self.set_loaded_snapshot(runtime_manager.loaded_model_ids());
        Ok(unloaded)
    }

    /// Returns the descriptors for models that are currently loaded in memory.
    pub async fn loaded_models(&self) -> Vec<nexo_core::ModelDescriptor> {
        let loaded_ids = self
            .loaded_model_ids
            .read()
            .ok()
            .map(|loaded| loaded.clone())
            .unwrap_or_default();
        loaded_ids
            .into_iter()
            .filter_map(|model_id| self.registry.get_model(&model_id))
            .collect()
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

    /// Submits a new inference request and returns a stream of responses.
    pub async fn submit(&self, request: InferenceRequest) -> nexo_core::Result<InferenceStream> {
        let (descriptor, runtime) = {
            let mut runtime_manager = self.runtime_manager.lock().await;
            let prepared = runtime_manager.prepare_request(&request).await;
            self.set_loaded_snapshot(runtime_manager.loaded_model_ids());
            prepared.map_err(Error::into_core_error)?
        };
        runtime
            .submit(descriptor, request)
            .await
            .map_err(Error::into_core_error)
    }
}

impl ModelRegistry for InferenceEngine {
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
