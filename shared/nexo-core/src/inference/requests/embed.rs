use crate::ModelSelection;
use crate::ReasoningSettings;
use crate::message::Conversation;
use crate::tools::ToolDefinition;
use serde::{Deserialize, Serialize};

/// A request for one or more embedding vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EmbedPayload {
    /// The ordered text inputs to embed.
    pub inputs: Vec<String>,
}

/// The input accepted by a tokenization request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum TokenizationInput {
    /// Raw text input.
    Text(String),

    /// A structured conversation input.
    Conversation(Conversation),
}

/// A request to tokenize text or conversation input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TokenizationPayload {
    /// The input to tokenize.
    pub input: TokenizationInput,

    /// The tools that should be considered during chat template tokenization.
    pub tools: Vec<ToolDefinition>,

    /// Whether a generation prompt should be appended.
    pub generation_prompt: GenerationPromptPolicy,

    /// Whether special tokens should be included.
    pub special_tokens: SpecialTokenPolicy,

    /// Reasoning controls that may affect prompt formatting.
    pub reasoning: ReasoningSettings,
}

/// Controls whether a generation prompt should be added during tokenization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum GenerationPromptPolicy {
    /// Do not add a generation prompt.
    #[default]
    Exclude,

    /// Add a generation prompt when tokenizing chat-style input.
    Include,
}

/// Controls whether special tokens are included during tokenization operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SpecialTokenPolicy {
    /// Include special tokens in the operation.
    #[default]
    Include,

    /// Exclude special tokens from the operation.
    Exclude,
}

/// A request to convert tokens back into textual content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetokenizationPayload {
    /// The tokens to detokenize.
    pub tokens: Vec<u32>,

    /// Whether special tokens should be included in the output text.
    pub special_tokens: SpecialTokenPolicy,
}
