use crate::{Error, Result};
use futures_util::{StreamExt, stream};
use nexo_core::{
    AudioFormat, GeneratedAudio, InferenceOperation, InferenceRequest, InferenceResponse,
    InferenceStream, MediaSource, ModelId, ModelRuntimeState, RequestId, SpeechGenerationPayload,
    SpeechGenerationResponse,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::warn;

/// Model Runtime for the AnyTTS inference engine.
pub(crate) struct AnyTtsRuntime {
    models: BTreeMap<ModelId, Arc<dyn any_tts::TtsModel>>,
}

impl AnyTtsRuntime {
    /// Creates a new AnyTtsRuntime with no pre-loaded models.
    pub(crate) fn new() -> Self {
        Self {
            models: BTreeMap::new(),
        }
    }

    /// Loads a model into the AnyTTS runtime.
    pub(crate) async fn load_model(&mut self, model_id: &ModelId) -> Result {
        if self.models.contains_key(model_id) {
            warn!(
                model_id = %model_id,
                "Model is already loaded in AnyTTS runtime"
            );
            return Ok(());
        }

        match model_id {
            ModelId::Kokoro82m => {
                let config = any_tts::TtsConfig::new(any_tts::ModelType::Kokoro)
                    .with_model_path("TODO")
                    .with_preferred_runtime();

                let model = any_tts::load_model(config)?;
                self.models.insert(model_id.clone(), Arc::from(model));
            }
            _ => {
                return Err(Error::UnsupportedFeature {
                    feature: format!("Model `{}` is not supported", model_id),
                });
            }
        }

        Ok(())
    }

    /// Unload a model from the AnyTTS runtime.
    pub(crate) async fn unload_model(&mut self, model_id: &ModelId) -> Result {
        match self.models.remove(model_id) {
            Some(_) => Ok(()),
            None => {
                warn!(model_id = %model_id, "Model is not loaded in AnyTTS runtime");
                Ok(())
            }
        }
    }

    /// Submits an inference request to the specified model in the AnyTTS runtime.
    pub(crate) async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        if !self.models.contains_key(model_id) {
            return Err(Error::ModelNotLoaded {
                model_id: model_id.clone(),
                current_state: ModelRuntimeState::Unloaded,
            });
        }

        match model_id {
            ModelId::Kokoro82m => match request.operation {
                InferenceOperation::GenerateSpeech(payload) => {
                    self.infer_speech_generation(model_id.clone(), request.request_id, payload)
                        .await
                }
                _ => Err(Error::UnsupportedRequest {
                    kind: "Kokoro82m only supports SpeechGeneration requests",
                }),
            },
            _ => {
                return Err(Error::UnsupportedFeature {
                    feature: format!("Model `{}` is not supported on AnyTts runtime", model_id),
                });
            }
        }
    }

    /// Performs speech generation inference using the specified model and request payload.
    async fn infer_speech_generation(
        &self,
        model_id: ModelId,
        request_id: RequestId,
        payload: SpeechGenerationPayload,
    ) -> Result<InferenceStream> {
        // Retrieve loaded model instance
        let model = self
            .models
            .get(&model_id)
            .ok_or_else(|| Error::ModelNotLoaded {
                model_id: model_id.clone(),
                current_state: ModelRuntimeState::Unloaded,
            })?
            .clone();

        // Perform speech inference on model in a blocking task
        let response = stream::once(async move {
            let result = match tokio::task::spawn_blocking(move || match model_id {
                ModelId::Kokoro82m => {
                    let mut synth_request = any_tts::SynthesisRequest::new(payload.text)
                        .with_language(payload.language);

                    if let Some(voice) = payload.voice {
                        synth_request = synth_request.with_voice(voice);
                    }

                    let audio = model.synthesize(&synth_request)?;

                    let generated_audio = GeneratedAudio {
                        source: MediaSource::Bytes(audio.get_wav()),
                        format: AudioFormat::Wav,
                        sample_rate_hz: Some(audio.sample_rate),
                        channel_count: Some(audio.channels),
                    };

                    Ok(InferenceResponse::Speech(SpeechGenerationResponse {
                        request_id,
                        model_id,
                        audio: generated_audio,
                    }))
                }
                _ => Err(Error::UnsupportedFeature {
                    feature: format!("Model `{}` is not supported on AnyTts runtime", model_id),
                }),
            })
            .await
            {
                Ok(inner) => inner,
                Err(e) => Err(Error::Runtime {
                    message: format!("Join error: {e}"),
                }),
            };

            result.map_err(|e| match e {
                other => nexo_core::Error::Inference {
                    message: format!("Inference error: {other}"),
                },
            })
        });

        Ok(response.boxed())
    }
}
