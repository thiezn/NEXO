use serde::{Deserialize, Serialize};

use crate::ModelCapability;
use crate::common::MetadataMap;
use crate::ids::{RequestId, RoundId, RunId, SessionId};
use crate::message::{
    AudioInput, ContentPart, Conversation, ConversationMessage, ImageInput, MediaSource,
    MessageRole, TextPart,
};
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
#[allow(clippy::large_enum_variant)]
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

impl GenerateRequest {
    /// Build a round-based conversational generation request.
    pub fn new_round(
        request_id: RequestId,
        session_id: SessionId,
        run_id: RunId,
        round_id: RoundId,
        model: ModelSelection,
        messages: Vec<ConversationMessage>,
        tools: Vec<ToolDefinition>,
        tool_choice: ToolChoice,
        reasoning: ReasoningSettings,
    ) -> Self {
        Self {
            request_id: Some(request_id),
            session_id: Some(session_id),
            run_id: Some(run_id),
            round_id: Some(round_id),
            model,
            conversation: Conversation {
                messages,
                metadata: MetadataMap::new(),
            },
            tools,
            tool_choice,
            reasoning,
            output_constraint: OutputConstraint::None,
            sampling: SamplingConfig::default(),
            streaming: StreamingMode::Buffered,
            metadata: MetadataMap::new(),
        }
    }

    /// Build a single-turn image analysis generation request.
    pub fn new_image_analyze(
        request_id: RequestId,
        image_data: String,
        media_type: Option<String>,
        prompt: String,
        max_output_tokens: usize,
        temperature: f32,
    ) -> Self {
        Self {
            request_id: Some(request_id),
            session_id: None,
            run_id: None,
            round_id: None,
            model: ModelSelection {
                specific_model: None,
                required_capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ImageInput,
                ],
                preferred_capabilities: Vec::new(),
            },
            conversation: Conversation {
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    parts: vec![
                        ContentPart::Image(ImageInput {
                            source: MediaSource::Base64(image_data),
                            media_type,
                        }),
                        ContentPart::Text(TextPart { text: prompt }),
                    ],
                    metadata: MetadataMap::new(),
                }],
                metadata: MetadataMap::new(),
            },
            tools: Vec::new(),
            tool_choice: ToolChoice::Disabled,
            reasoning: ReasoningSettings::default(),
            output_constraint: OutputConstraint::None,
            sampling: SamplingConfig {
                max_output_tokens: Some(max_output_tokens),
                temperature: Some(temperature),
                ..SamplingConfig::default()
            },
            streaming: StreamingMode::Buffered,
            metadata: MetadataMap::new(),
        }
    }

    /// Build a single-turn audio analysis generation request.
    pub fn new_audio_analyze(
        request_id: RequestId,
        audio_data: String,
        media_type: Option<String>,
        sample_rate_hz: Option<u32>,
        channel_count: Option<u16>,
        prompt: String,
        max_output_tokens: usize,
        temperature: f32,
    ) -> Self {
        Self {
            request_id: Some(request_id),
            session_id: None,
            run_id: None,
            round_id: None,
            model: ModelSelection {
                specific_model: None,
                required_capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::AudioInput,
                ],
                preferred_capabilities: Vec::new(),
            },
            conversation: Conversation {
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    parts: vec![
                        ContentPart::Audio(AudioInput {
                            source: MediaSource::Base64(audio_data),
                            media_type,
                            sample_rate_hz,
                            channel_count,
                        }),
                        ContentPart::Text(TextPart { text: prompt }),
                    ],
                    metadata: MetadataMap::new(),
                }],
                metadata: MetadataMap::new(),
            },
            tools: Vec::new(),
            tool_choice: ToolChoice::Disabled,
            reasoning: ReasoningSettings::default(),
            output_constraint: OutputConstraint::None,
            sampling: SamplingConfig {
                max_output_tokens: Some(max_output_tokens),
                temperature: Some(temperature),
                ..SamplingConfig::default()
            },
            streaming: StreamingMode::Buffered,
            metadata: MetadataMap::new(),
        }
    }
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn new_image_analyze_builds_expected_message() {
        let request = GenerateRequest::new_image_analyze(
            RequestId::from("req-1"),
            "abcd".to_string(),
            Some("image/png".to_string()),
            "describe this image".to_string(),
            512,
            0.3,
        );

        assert_eq!(request.request_id, Some(RequestId::from("req-1")));
        assert_eq!(request.model.required_capabilities.len(), 2);
        assert_eq!(request.sampling.max_output_tokens, Some(512));
        assert_eq!(request.sampling.temperature, Some(0.3));

        let message = &request.conversation.messages[0];
        assert!(matches!(message.role, MessageRole::User));
        assert!(matches!(message.parts[0], ContentPart::Image(_)));
        assert!(matches!(message.parts[1], ContentPart::Text(_)));
    }

    #[test]
    fn new_audio_analyze_builds_expected_message() {
        let request = GenerateRequest::new_audio_analyze(
            RequestId::from("req-2"),
            "efgh".to_string(),
            Some("audio/wav".to_string()),
            Some(16_000),
            Some(1),
            "summarize this audio".to_string(),
            1024,
            0.8,
        );

        assert_eq!(request.request_id, Some(RequestId::from("req-2")));
        assert_eq!(request.model.required_capabilities.len(), 2);
        assert_eq!(request.sampling.max_output_tokens, Some(1024));
        assert_eq!(request.sampling.temperature, Some(0.8));

        let message = &request.conversation.messages[0];
        assert!(matches!(message.role, MessageRole::User));
        assert!(matches!(message.parts[0], ContentPart::Audio(_)));
        assert!(matches!(message.parts[1], ContentPart::Text(_)));
    }
}
