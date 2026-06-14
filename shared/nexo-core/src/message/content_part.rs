use serde::{Deserialize, Serialize};

use crate::tools::{ToolCall, ToolResult};

use super::{AudioInput, ImageInput, VideoInput};

/// A single multimodal or structured part of a conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain textual content.
    Text(String),

    /// An image input.
    Image(ImageInput),

    /// A video input.
    Video(VideoInput),

    /// An audio input.
    Audio(AudioInput),

    /// A model-emitted tool call.
    ToolCall(ToolCall),

    /// A tool result injected into the conversation.
    ToolResult(ToolResult),
}
