use serde::{Deserialize, Serialize};

/// A discrete capability that a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    /// The model can generate text responses from conversational input.
    TextGeneration,

    /// The model can emit structured tool call requests.
    ToolCalling,

    /// The model can produce embedding vectors from text input.
    Embeddings,

    /// The model can consume image inputs as part of a generation request.
    ImageInput,

    /// The model can consume video inputs as part of a generation request.
    VideoInput,

    /// The model can consume audio inputs as part of a generation request.
    AudioInput,

    /// The model can generate images from textual prompts.
    ImageGeneration,

    /// The model can synthesize speech from textual prompts.
    SpeechGeneration,

    /// The model can enforce structured output constraints.
    StructuredOutput,

    /// The model supports explicit reasoning controls such as effort tuning.
    Reasoning,

    /// The model can emit partial streamed responses.
    Streaming,
}
