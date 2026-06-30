use super::{
    DetokenizationPayload, EmbedPayload, ImageGenerationPayload, MultiModalPayload,
    SpeechGenerationPayload, TokenizationPayload,
};
use crate::{Error, ModelDefinition, ModelSelection, Result};
use crate::{ModelId, OperationId, RoundId, RunId, SessionId};
use serde::{Deserialize, Serialize};

/// A unified request struct for all supported inference operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct InferenceRequest {
    /// The unique identifier for the inference request.
    pub operation_id: OperationId,

    /// The session identifier for the request, used to correlate related requests.
    pub session_id: SessionId,

    /// The run identifier used by higher-level orchestration to correlate requests
    /// across a workflow.
    pub run_id: RunId,

    /// The round identifier used to correlate a single loop iteration.
    pub round_id: RoundId,

    /// The model selection criteria for the request.
    pub model_selection: ModelSelection,

    /// The specific inference operation being requested.
    #[serde(flatten)]
    pub operation: InferenceOperation,
}

impl InferenceRequest {
    /// Returns the ModelId based on the model selection criteria and available model definitions.
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
                    Err(Error::ModelNotFound {
                        model_id: model_id.clone(),
                    })
                }
            }
            ModelSelection::Capabilities(_capabilities) => {
                todo!("Find the first loaded model that satisfies the required capabilities")
            }
        }
    }

    /// Returns the kind of inference operation carried by this request.
    pub fn operation_kind(&self) -> InferenceOperationKind {
        self.operation.kind()
    }
}

/// The specific inference operation being requested, with associated payload.
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema, strum::EnumDiscriminants,
)]
#[strum_discriminants(name(InferenceOperationKind))]
#[strum_discriminants(vis(pub))]
#[strum_discriminants(doc = "The category of inference operation being executed.")]
#[strum_discriminants(derive(
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr
))]
#[strum_discriminants(serde(rename_all = "snake_case"))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
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

impl InferenceOperation {
    /// Returns the operation kind for this request payload.
    pub fn kind(&self) -> InferenceOperationKind {
        self.into()
    }
}
