use crate::{Error, Result};
use futures_util::{StreamExt, stream};
use mold_ai_inference::Flux2Engine;
use nexo_core::{
    AudioFormat, GeneratedAudio, InferenceOperation, InferenceRequest, InferenceResponse,
    InferenceStream, MediaSource, ModelId, RequestId, SpeechGenerationPayload,
    SpeechGenerationResponse,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Model Runtime for the Mold inference engine.
pub(crate) struct MoldRuntime {
    models: BTreeMap<ModelId, Arc<Flux2Engine>>,
}

impl MoldRuntime {
    /// Creates a new MoldRuntime with no pre-loaded models.
    pub(crate) fn new() -> Self {
        // TODO: pre-create all the Flux2Engine instances for each model.
        // Our load and unload operations will then just call load() and unload()
        // on the relevant Flux2Engine instance.
        Self {
            models: BTreeMap::new(),
        }
    }

    /// Loads a model into the Mold runtime.
    pub(crate) async fn load_model(&self, model_id: &ModelId) -> Result {
        todo!("Implement model loading for Mold runtime");

        // A Flux2Engine instance is bound for a specific model.
        // Initializing it doesn't load the model into memory, for that we need to call flux2_engine.load()
        // let flux2_engine = Flux2Engine::new(
        //     model_id.into(),
        //     build_flux2_paths(loader)?,
        //     runtime_config.qwen3_variant.clone(),
        //     map_load_strategy(runtime_config.load_strategy),
        //     runtime_config.gpu_ordinal,
        //     runtime_config.offload,
        //     None,
        // );

        Ok(())
    }

    /// Unload a model from the Mold runtime.
    pub(crate) async fn unload_model(&self, model_id: &ModelId) -> Result {
        todo!("Implement model unloading for Mold runtime");
    }

    /// Submits an inference request to the specified model in the Mold runtime.
    pub(crate) async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        todo!("Implement inference request submission for Mold runtime");
    }
}
