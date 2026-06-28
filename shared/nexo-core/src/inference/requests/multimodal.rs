use crate::ReasoningSettings;
use crate::message::{
    AudioInput, ContentPart, Conversation, ConversationMessage, ImageInput, MediaSource,
    MessageRole,
};
use crate::tools::{ToolChoice, ToolDefinition};
use crate::{OutputConstraint, SamplingConfig, StreamingMode};
use serde::{Deserialize, Serialize};

/// A request for text or multimodal conversational generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MultiModalPayload {
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
}

impl MultiModalPayload {
    /// Build a round-based conversational generation request.
    pub fn new_round(
        messages: Vec<ConversationMessage>,
        tools: Vec<ToolDefinition>,
        tool_choice: ToolChoice,
        reasoning: ReasoningSettings,
    ) -> Self {
        Self {
            conversation: Conversation { messages },
            tools,
            tool_choice,
            reasoning,
            output_constraint: OutputConstraint::None,
            sampling: SamplingConfig::default(),
            streaming: StreamingMode::Buffered,
        }
    }

    /// Build a single-turn image analysis generation request.
    pub fn new_image_analyze(
        image_data: String,
        media_type: Option<String>,
        prompt: String,
        max_output_tokens: usize,
        temperature: f32,
    ) -> Self {
        Self {
            conversation: Conversation {
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    parts: vec![
                        ContentPart::Image(ImageInput {
                            source: MediaSource::Base64(image_data),
                            media_type,
                        }),
                        ContentPart::Text(prompt),
                    ],
                }],
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
        }
    }

    /// Build a single-turn audio analysis generation request.
    pub fn new_audio_analyze(
        audio_data: String,
        media_type: Option<String>,
        sample_rate_hz: Option<u32>,
        channel_count: Option<u16>,
        prompt: String,
        max_output_tokens: usize,
        temperature: f32,
    ) -> Self {
        Self {
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
                        ContentPart::Text(prompt),
                    ],
                }],
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
        }
    }
}
