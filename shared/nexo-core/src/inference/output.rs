use serde::{Deserialize, Serialize};

use super::InferenceOperationKind;
use super::responses::{
    DetokenizationResponse, EmbedResponse, ImageGenerationResponse, MultiModalResponse,
    SpeechGenerationResponse, TokenizationResponse,
};

/// Successful full output from an inference operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceOutput {
    /// Final conversational or multimodal generation output.
    MultiModal(MultiModalResponse),

    /// Final embedding vectors.
    Embed(EmbedResponse),

    /// Final generated image artifacts.
    GenerateImage(ImageGenerationResponse),

    /// Final generated speech artifact.
    GenerateSpeech(SpeechGenerationResponse),

    /// Final tokenized output.
    Tokenize(TokenizationResponse),

    /// Final detokenized output.
    Detokenize(DetokenizationResponse),
}

impl InferenceOutput {
    /// Returns the operation kind that produced this final output.
    pub const fn operation_kind(&self) -> InferenceOperationKind {
        match self {
            Self::MultiModal(_) => InferenceOperationKind::MultiModal,
            Self::Embed(_) => InferenceOperationKind::Embed,
            Self::GenerateImage(_) => InferenceOperationKind::GenerateImage,
            Self::GenerateSpeech(_) => InferenceOperationKind::GenerateSpeech,
            Self::Tokenize(_) => InferenceOperationKind::Tokenize,
            Self::Detokenize(_) => InferenceOperationKind::Detokenize,
        }
    }
}
