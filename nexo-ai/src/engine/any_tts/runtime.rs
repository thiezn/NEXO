use std::sync::Arc;

use any_tts::{ModelType, SynthesisRequest, TtsConfig, TtsModel, load_model};
use futures_util::{StreamExt, stream};
use nexo_core::inference::request::{GeneratedAudio, SpeechGenerationRequest};
use nexo_core::{
    InferenceErrorCode, InferenceFailure, InferenceRequest, InferenceResponse, InferenceStream,
    MediaSource, ModelDescriptor, ModelId, RequestId, Retryability, SpeechLanguage,
};
use nexo_model_mgmt::resolve_model_storage_dir;
use serde_json::Value;

use crate::{Error, RegisteredModelConfig, Result};

pub(crate) const INTERNAL_RUNTIME_KEY: &str = "internal_runtime";
pub(crate) const KOKORO_RUNTIME_ID: &str = "any_tts_kokoro";

type SharedTtsModel = Arc<dyn TtsModel>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InternalRuntimeKind {
    Kokoro,
}

#[derive(Clone)]
pub(crate) struct AnyTtsRuntime {
    model: SharedTtsModel,
}

impl std::fmt::Debug for AnyTtsRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyTtsRuntime").finish_non_exhaustive()
    }
}

impl AnyTtsRuntime {
    pub(crate) async fn from_model_config(model: &RegisteredModelConfig) -> Result<Self> {
        match internal_runtime_kind(model) {
            Some(InternalRuntimeKind::Kokoro) => load_kokoro_model(model).await,
            None => Err(Error::UnsupportedFeature {
                feature: format!(
                    "model `{}` is not configured for a private any-tts runtime",
                    model.descriptor.id
                ),
            }),
        }
    }

    pub(crate) async fn submit(
        &self,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match request {
            InferenceRequest::GenerateSpeech(request) => {
                self.submit_generate_speech(descriptor, request).await
            }
            other => Err(Error::UnsupportedRequest {
                kind: request_kind(&other),
            }),
        }
    }

    async fn submit_generate_speech(
        &self,
        descriptor: ModelDescriptor,
        request: SpeechGenerationRequest,
    ) -> Result<InferenceStream> {
        let model = self.model.clone();
        let request_id = request.request_id.clone();
        let model_id = descriptor.id.clone();

        Ok(stream::once(async move {
            let response = tokio::task::spawn_blocking(move || {
                let synth_request = map_speech_request(&request);
                let audio = model
                    .synthesize(&synth_request)
                    .map_err(map_any_tts_error)?;
                Ok::<_, Error>(map_speech_generation_response(request.request_id, model_id, audio))
            })
            .await;

            Ok(match response {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => map_failure_response(error, request_id),
                Err(error) => map_failure_response(
                    Error::Runtime {
                        message: format!("failed to join any-tts request task: {error}"),
                    },
                    request_id,
                ),
            })
        })
        .boxed())
    }
}

pub(crate) fn internal_runtime_kind(model: &RegisteredModelConfig) -> Option<InternalRuntimeKind> {
    match model
        .descriptor
        .metadata
        .get(INTERNAL_RUNTIME_KEY)
        .and_then(Value::as_str)
    {
        Some(KOKORO_RUNTIME_ID) => Some(InternalRuntimeKind::Kokoro),
        _ => None,
    }
}

async fn load_kokoro_model(model: &RegisteredModelConfig) -> Result<AnyTtsRuntime> {
    let storage_reference = model_storage_reference(&model.descriptor);

    tokio::task::spawn_blocking(move || {
        let model_dir = resolve_model_storage_dir(&storage_reference);
        let config = TtsConfig::new(ModelType::Kokoro)
            .with_model_path(model_dir.to_string_lossy().into_owned())
            .with_preferred_runtime();
        let model = load_model(config).map_err(map_any_tts_error)?;
        Ok(AnyTtsRuntime {
            model: Arc::<dyn TtsModel>::from(model),
        })
    })
    .await
    .map_err(|error| Error::Runtime {
        message: format!("failed to join any-tts model-load task: {error}"),
    })?
}

fn model_storage_reference(descriptor: &ModelDescriptor) -> String {
    descriptor
        .metadata
        .get("source_model")
        .and_then(Value::as_str)
        .unwrap_or_else(|| descriptor.id.as_str())
        .to_string()
}

fn map_speech_request(request: &SpeechGenerationRequest) -> SynthesisRequest {
    let mut synth_request = SynthesisRequest::new(request.text.clone())
        .with_language(language_code(request.language));

    if let Some(voice) = &request.voice {
        synth_request = synth_request.with_voice(voice.clone());
    }

    synth_request
}

fn language_code(language: SpeechLanguage) -> &'static str {
    match language {
        SpeechLanguage::English => "en",
        SpeechLanguage::Dutch => "nl",
    }
}

fn map_speech_generation_response(
    request_id: Option<RequestId>,
    model_id: ModelId,
    audio: any_tts::AudioSamples,
) -> InferenceResponse {
    let audio = GeneratedAudio {
        source: MediaSource::Bytes(audio.get_wav()),
        format: nexo_core::AudioFormat::Wav,
        sample_rate_hz: Some(audio.sample_rate),
        channel_count: Some(audio.channels),
    };

    InferenceResponse::Speech(nexo_core::SpeechGenerationResponse {
        request_id,
        model_id: Some(model_id),
        audio,
    })
}

fn map_failure_response(error: Error, request_id: Option<RequestId>) -> InferenceResponse {
    let (code, retryability) = match error {
        Error::UnsupportedRequest { .. } | Error::UnsupportedFeature { .. } => {
            (InferenceErrorCode::UnsupportedFeature, Retryability::Fatal)
        }
        Error::UnknownModel { .. } | Error::ModelNotLoaded { .. } => {
            (InferenceErrorCode::ModelUnavailable, Retryability::Fatal)
        }
        Error::Json(_) | Error::Core(_) => {
            (InferenceErrorCode::InvalidRequest, Retryability::Fatal)
        }
        Error::EmptyModelCatalog
        | Error::DuplicateModelId { .. }
        | Error::UnresolvedModelSelection { .. }
        | Error::UnsupportedMessagePart { .. }
        | Error::InvalidToolPayload { .. }
        | Error::Config { .. }
        | Error::Runtime { .. }
        | Error::Io(_) => (InferenceErrorCode::Internal, Retryability::Retryable),
    };

    InferenceResponse::Failure(InferenceFailure {
        request_id,
        run_id: None,
        round_id: None,
        code,
        message: error.to_string(),
        retryability,
    })
}

fn map_any_tts_error(error: any_tts::TtsError) -> Error {
    Error::Runtime {
        message: error.to_string(),
    }
}

fn request_kind(request: &InferenceRequest) -> &'static str {
    match request {
        InferenceRequest::Generate(_) => "generate",
        InferenceRequest::Embed(_) => "embed",
        InferenceRequest::GenerateImage(_) => "generate_image",
        InferenceRequest::GenerateSpeech(_) => "generate_speech",
        InferenceRequest::Tokenize(_) => "tokenize",
        InferenceRequest::Detokenize(_) => "detokenize",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn request_language_prefers_speech_specific_metadata() {
        let request = SpeechGenerationRequest {
            request_id: None,
            session_id: None,
            model: nexo_core::ModelSelection::default(),
            text: "hello".to_string(),
            language: SpeechLanguage::Dutch,
            voice: None,
            format: nexo_core::AudioFormat::Wav,
            sample_rate_hz: None,
            speed: None,
            metadata: Default::default(),
        };

        assert_eq!(language_code(request.language), "nl");
    }

    #[test]
    fn internal_runtime_kind_detects_kokoro_backend() {
        let mut descriptor = ModelDescriptor {
            id: "kokoro-82m-tts".into(),
            display_name: "Kokoro 82M TTS".to_string(),
            provider: Some("hexgrad".to_string()),
            runtime: nexo_core::InferenceRuntime::AnyTts,
            capabilities: vec![nexo_core::ModelCapability::SpeechGeneration],
            modalities: nexo_core::ModelModalities {
                input: vec![nexo_core::SupportedModality::Text],
                output: vec![nexo_core::SupportedModality::Audio],
            },
            role_strategy: nexo_core::RoleStrategy::Default,
            context_window_tokens: None,
            max_output_tokens: None,
            metadata: Default::default(),
        };
        descriptor.metadata.insert(
            INTERNAL_RUNTIME_KEY.to_string(),
            Value::String(KOKORO_RUNTIME_ID.to_string()),
        );

        assert_eq!(
            internal_runtime_kind(&RegisteredModelConfig {
                descriptor,
                runtimes: Vec::new(),
            }),
            Some(InternalRuntimeKind::Kokoro)
        );
    }
}
