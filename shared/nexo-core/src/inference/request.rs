use serde::{Deserialize, Serialize};

use crate::common::MetadataMap;
use crate::ids::{RequestId, RoundId, RunId, SessionId};
use crate::message::{Conversation, MediaSource};
use crate::model::{ModelSelection, ReasoningSettings};
use crate::tools::{ToolChoice, ToolDefinition};

use super::{OutputConstraint, SamplingConfig, StreamingMode};

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

/// The requested audio encoding for synthesized speech output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// Raw pulse-code modulation samples.
    Pcm,

    /// WAV-encoded audio.
    Wav,

    /// MP3-encoded audio.
    Mp3,
}

/// A unified request enum for all supported inference operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceRequest {
    /// Generate text or multimodal conversational output.
    Generate(GenerateRequest),

    /// Produce embedding vectors for text input.
    Embed(EmbedRequest),

    /// Generate one or more images from a prompt.
    GenerateImage(ImageGenerationRequest),

    /// Generate speech audio from text input.
    GenerateSpeech(SpeechGenerationRequest),

    /// Tokenize raw text or a conversation.
    Tokenize(TokenizationRequest),

    /// Convert tokens back into text.
    Detokenize(DetokenizationRequest),
}

/// A request for text or multimodal conversational generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GenerateRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The optional session identifier used for continuity.
    pub session_id: Option<SessionId>,

    /// The optional run identifier used by higher-level orchestration.
    pub run_id: Option<RunId>,

    /// The optional round identifier used to correlate a single loop iteration.
    pub round_id: Option<RoundId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

    /// The conversation submitted to the model.
    pub conversation: Conversation,

    /// The tools exposed to the model during generation.
    pub tools: Vec<ToolDefinition>,

    /// The tool use policy applied during generation.
    pub tool_choice: ToolChoice,

    /// The requested reasoning controls for the generation.
    pub reasoning: ReasoningSettings,

    /// Structured output constraints applied to the response.
    pub output_constraint: OutputConstraint,

    /// Sampling and decoding parameters for the request.
    pub sampling: SamplingConfig,

    /// Whether the response should be buffered or streamed.
    pub streaming: StreamingMode,

    /// Additional request metadata.
    pub metadata: MetadataMap,
}

/// A request for one or more embedding vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EmbedRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The optional session identifier used for continuity.
    pub session_id: Option<SessionId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

    /// The ordered text inputs to embed.
    pub inputs: Vec<String>,

    /// Additional request metadata.
    pub metadata: MetadataMap,
}

/// The desired image size for an image generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImageGenerationSize {
    /// The image width, in pixels.
    pub width: u32,

    /// The image height, in pixels.
    pub height: u32,
}

/// A request to generate one or more images from text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImageGenerationRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The optional session identifier used for continuity.
    pub session_id: Option<SessionId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

    /// The positive prompt used for generation.
    pub prompt: String,

    /// The negative prompt used to suppress unwanted attributes.
    pub negative_prompt: Option<String>,

    /// The generated image size.
    pub size: ImageGenerationSize,

    /// The number of images requested.
    pub sample_count: u32,

    /// The diffusion step count, if supported by the runtime.
    pub steps: Option<u32>,

    /// The guidance scale, if supported by the runtime.
    pub guidance_scale: Option<f32>,

    /// The deterministic random seed, if requested.
    pub seed: Option<u64>,

    /// Additional request metadata.
    pub metadata: MetadataMap,
}

/// A request to synthesize speech from text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SpeechGenerationRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The optional session identifier used for continuity.
    pub session_id: Option<SessionId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

    /// The text to synthesize into speech.
    pub text: String,

    /// The requested voice label, if the runtime supports voice selection.
    pub voice: Option<String>,

    /// The desired audio format for the output.
    pub format: AudioFormat,

    /// The desired output sample rate, in hertz, if supported.
    pub sample_rate_hz: Option<u32>,

    /// The requested speaking speed multiplier, if supported.
    pub speed: Option<f32>,

    /// Additional request metadata.
    pub metadata: MetadataMap,
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
pub struct TokenizationRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

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

    /// Additional request metadata.
    pub metadata: MetadataMap,
}

/// A request to convert tokens back into textual content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetokenizationRequest {
    /// The optional request identifier supplied by the caller.
    pub request_id: Option<RequestId>,

    /// The model selection criteria for the request.
    pub model: ModelSelection,

    /// The tokens to detokenize.
    pub tokens: Vec<u32>,

    /// Whether special tokens should be included in the output text.
    pub special_tokens: SpecialTokenPolicy,

    /// Additional request metadata.
    pub metadata: MetadataMap,
}

/// A transport-safe generated image payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GeneratedImage {
    /// The zero-based order of the image within the response.
    pub index: usize,

    /// The generated image content.
    pub source: MediaSource,

    /// The optional media type of the generated image.
    pub media_type: Option<String>,

    /// The generated image width, in pixels, if known.
    pub width: Option<u32>,

    /// The generated image height, in pixels, if known.
    pub height: Option<u32>,
}

/// A transport-safe generated audio payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GeneratedAudio {
    /// The generated audio content.
    pub source: MediaSource,

    /// The format used for the generated audio.
    pub format: AudioFormat,

    /// The audio sample rate, in hertz, if known.
    pub sample_rate_hz: Option<u32>,

    /// The number of audio channels, if known.
    pub channel_count: Option<u16>,
}
