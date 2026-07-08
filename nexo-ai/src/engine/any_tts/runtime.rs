use crate::catalog::ModelManifest;
use crate::{Error, Result};
use futures_util::{StreamExt, stream};
use nexo_core::{
    AudioFormat, GeneratedAudio, InferenceMeta, InferenceOperation, InferenceOutput,
    InferenceRequest, InferenceStream, InferenceUpdate, MediaSource, ModelId, ModelRuntimeState,
    SpeechGenerationPayload, SpeechGenerationResponse,
};
use std::sync::Arc;
use tracing::warn;

/// Model Runtime for the AnyTTS inference engine.
pub(crate) struct AnyTtsRuntime {
    /// The manifest that defines the model being loaded into this runtime instance.
    manifest: ModelManifest,

    /// The live AnyTTS model once loaded.
    model: Option<Arc<dyn any_tts::TtsModel>>,
}

impl AnyTtsRuntime {
    /// Creates a new unloaded AnyTTS runtime for the provided model manifest.
    pub(crate) fn new(manifest: ModelManifest) -> Self {
        Self {
            manifest,
            model: None,
        }
    }

    /// Loads a model into the AnyTTS runtime.
    pub(crate) async fn load_model(&mut self, model_id: &ModelId) -> Result {
        if self.model.is_some() {
            warn!(
                model_id = %model_id,
                "Model is already loaded in AnyTTS runtime"
            );
            return Ok(());
        }

        if self.manifest.model_id() != model_id {
            return Err(Error::UnsupportedFeature {
                feature: format!(
                    "AnyTTS runtime for `{}` cannot load model `{}`",
                    self.manifest.model_id(),
                    model_id,
                ),
            });
        }

        match model_id {
            ModelId::Kokoro82m => {
                let model_dir = self.manifest.model_dir()?;
                let config = any_tts::TtsConfig::new(any_tts::ModelType::Kokoro)
                    .with_model_path(model_dir.to_string_lossy().into_owned())
                    .with_preferred_runtime();

                let model = any_tts::load_model(config)?;
                self.model = Some(Arc::from(model));
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
        if self.model.take().is_none() {
            warn!(model_id = %model_id, "Model is not loaded in AnyTTS runtime");
        }

        Ok(())
    }

    /// Submits an inference request to the specified model in the AnyTTS runtime.
    pub(crate) async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        if self.model.is_none() {
            return Err(Error::ModelNotLoaded {
                model_id: model_id.clone(),
                current_state: ModelRuntimeState::Unloaded,
            });
        }

        if self.manifest.model_id() != model_id {
            return Err(Error::UnsupportedFeature {
                feature: format!(
                    "AnyTTS runtime for `{}` cannot run inference on model `{}`",
                    self.manifest.model_id(),
                    model_id,
                ),
            });
        }

        match model_id {
            ModelId::Kokoro82m => {
                let meta = InferenceMeta::from_request(&request);
                match request.operation {
                    InferenceOperation::GenerateSpeech(payload) => {
                        self.infer_speech_generation(meta, payload).await
                    }
                    _ => Err(Error::UnsupportedRequest {
                        kind: "Kokoro82m only supports SpeechGeneration requests",
                    }),
                }
            }
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
        meta: InferenceMeta,
        payload: SpeechGenerationPayload,
    ) -> Result<InferenceStream> {
        // Retrieve loaded model instance
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::ModelNotLoaded {
                model_id: meta.model_id.clone(),
                current_state: ModelRuntimeState::Unloaded,
            })?
            .clone();

        // Perform speech inference on model in a blocking task
        let response = stream::once(async move {
            let result = match tokio::task::spawn_blocking(move || {
                let mut synth_request =
                    any_tts::SynthesisRequest::new(payload.text).with_language(payload.language);

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

                let speech = SpeechGenerationResponse {
                    audio: generated_audio,
                };

                Ok(InferenceUpdate::completed(
                    meta,
                    InferenceOutput::GenerateSpeech(speech),
                ))
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
