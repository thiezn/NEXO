use super::{
    DetokenizationPayload, EmbedPayload, ImageGenerationPayload, MultiModalPayload,
    SpeechGenerationPayload, TokenizationPayload,
};
use crate::{Error, ModelDefinition, ModelSelection, Result};
use crate::{ModelId, OperationId, SessionId};
use serde::{Deserialize, Serialize};

#[cfg(feature = "sqlx")]
use sqlx::error::BoxDynError;
#[cfg(feature = "sqlx")]
use sqlx::sqlite::{Sqlite, SqliteValueRef};
#[cfg(feature = "sqlx")]
use sqlx::{Decode, Type};

/// A unified representation of request for inference by a user.
///
/// - The type of inference request can be any modality
/// - It only conveys the intent of the user, and does not include any system prompt or
///   model selection logic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct InferenceIntent {
    /// The unique identifier for the inference request.
    pub operation_id: OperationId,

    /// The session identifier for the request, used to correlate related requests.
    pub session_id: SessionId,

    /// The model selection criteria for the request.
    pub model_selection: ModelSelection,

    /// The specific inference operation being requested.
    #[serde(flatten)]
    pub operation: InferenceOperation,
}

impl InferenceIntent {
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

#[cfg(feature = "sqlx")]
impl Type<Sqlite> for InferenceOperationKind {
    fn type_info() -> <Sqlite as sqlx::Database>::TypeInfo {
        <String as Type<Sqlite>>::type_info()
    }

    fn compatible(ty: &<Sqlite as sqlx::Database>::TypeInfo) -> bool {
        <String as Type<Sqlite>>::compatible(ty)
    }
}

#[cfg(feature = "sqlx")]
impl<'r> Decode<'r, Sqlite> for InferenceOperationKind {
    fn decode(value: SqliteValueRef<'r>) -> Result<Self, BoxDynError> {
        let value = <String as Decode<Sqlite>>::decode(value)?;
        value.parse().map_err(Box::<dyn std::error::Error + Send + Sync>::from)
    }
}
