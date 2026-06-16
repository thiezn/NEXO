use super::{
    DetokenizationPayload, EmbedPayload, ImageGenerationPayload, MultiModalPayload,
    SpeechGenerationPayload, TokenizationPayload,
};
use crate::common::MetadataMap;
use crate::ids::{ModelId, RequestId, RoundId, RunId, SessionId};
use crate::{ModelDefinition, ModelSelection, Result};
use serde::{Deserialize, Serialize};

/// A unified request struct for all supported inference operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct InferenceRequest {
    /// The unique identifier for the inference request.
    pub request_id: RequestId,

    /// The session identifier for the request, used to correlate related requests.
    pub session_id: SessionId,

    /// The run identifier used by higher-level orchestration to correlate requests
    /// across a workflow.
    pub run_id: RunId,

    /// The round identifier used to correlate a single loop iteration.
    pub round_id: RoundId,

    /// The model selection criteria for the request.
    pub model_selection: ModelSelection,

    /// Additional metadata associated with the request.
    pub metadata: MetadataMap,

    /// The specific inference operation being requested.
    #[serde(flatten)]
    pub operation: InferenceOperation,
}

impl InferenceRequest {
    /// Returns the ModelId
    ///
    /// # Arguments
    /// * `available_model_definitions` - A list of all available model definitions to consider for capability-based selection.
    pub fn model(&self, available_model_definitions: Vec<&ModelDefinition>) -> Result<&ModelId> {
        match &self.model_selection {
            ModelSelection::SpecificModel(model_id) => {
                if let Some(_) = available_model_definitions
                    .iter()
                    .find(|def| def.id() == model_id)
                {
                    Ok(model_id)
                } else {
                    todo!("Raise model not known error");
                }
            }
            ModelSelection::Capabilities(_capabilities) => {
                todo!("Find the first loaded model that satisfies the required capabilities")
            }
        }
    }
}

/// The specific inference operation being requested, with associated payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceOperation {
    /// Generate text or multimodal conversational output.
    MultiModal(MultiModalPayload),

    /// Produce embedding vectors for text input.
    Embed(EmbedPayload),

    /// Generate one or more images from a prompt.
    GenerateImage(ImageGenerationPayload),

    /// Generate speech audio from text input.
    GenerateSpeech(SpeechGenerationPayload),

    /// Tokenize raw text or a conversation.
    Tokenize(TokenizationPayload),

    /// Convert tokens back into text.
    Detokenize(DetokenizationPayload),
}
