use crate::{Error, Result};
use futures_util::{StreamExt, stream};
use mistralrs_core::MistralRsBuilder;
use mold_ai_inference::Flux2Engine;
use nexo_core::{
    AudioFormat, GeneratedAudio, InferenceOperation, InferenceRequest, InferenceResponse,
    InferenceStream, MediaSource, ModelId, RequestId, SpeechGenerationPayload,
    SpeechGenerationResponse,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::warn;

/// Model Runtime for the MistralRs inference engine.
pub(crate) struct MistralRsRuntime {
    /// The loaded MistralRs runtime, if it has been initialized.
    ///
    /// NOTE: The MistralRs does not seem to support loading it
    /// without actually loading a model in memory.
    runtime: Arc<mistralrs_core::MistralRs>,
}

impl MistralRsRuntime {
    pub fn new() -> Self {
        todo!("Implement MistralRs runtime initialization");

        // // TODO: Initialize pipelines for each model.
        // let pipelines: Vec<mistralrs_core::Pipeline> = Vec::new();

        // let runtime = MistralRsBuilder::new(pipelines.first(), scheduler, true, None)
        //     .with_model_id(first.descriptor.id.to_string())
        //     .with_no_kv_cache(runtime_config.no_kv_cache)
        //     .with_no_prefix_cache(runtime_config.no_prefix_cache)
        //     .with_prefix_cache_n(runtime_config.prefix_cache_entries)
        //     .with_disable_eos_stop(runtime_config.disable_eos_stop)
        //     .build()
        //     .await;

        // Self { runtime }
    }

    /// Loads a model into the MistralRs runtime.
    pub(crate) async fn load_model(&self, model_id: &ModelId) -> Result {
        todo!("Implement model loading for MistralRs runtime");
    }

    /// Unload a model from the MistralRs runtime.
    pub(crate) async fn unload_model(&self, model_id: &ModelId) -> Result {
        todo!("Implement model unloading for MistralRs runtime");
    }

    /// Submits an inference request to the specified model in the MistralRs runtime.
    pub(crate) async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        todo!("Implement inference request submission for MistralRs runtime");
    }
}
