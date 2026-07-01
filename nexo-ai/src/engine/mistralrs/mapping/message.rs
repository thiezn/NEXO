use crate::engine::mistralrs::mapping::media::{map_audio_input, map_image_input};
use crate::engine::mistralrs::mapping::tools::{
    map_role, map_tool_calls_field, map_tool_result_message, serialize_tool_calls,
};
use crate::{Error, Result};
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{AudioInput as MistralAudioInput, MessageContent};
use nexo_core::{
    ContentPart, ConversationMessage, MessageRole, RoleStrategy,
};

/// Aggregates the mapped chat transcript alongside the referenced media payloads.
#[derive(Default)]
pub(crate) struct MessageMapping {
    /// Ordered text or multimodal messages encoded for Mistral.rs.
    pub(crate) messages: Vec<IndexMap<String, MessageContent>>,

    /// Ordered image payloads referenced by the mapped messages.
    pub(crate) images: Vec<image::DynamicImage>,

    /// Ordered audio payloads referenced by the mapped messages.
    pub(crate) audios: Vec<MistralAudioInput>,
}

/// Maps one shared conversation message into one or more Mistral.rs message records.
///
/// # Arguments
///
/// * `message` - The shared conversation message being translated for Mistral.rs.
/// * `role_strategy` - The model-specific role mapping strategy to apply.
/// * `images` - The ordered image payload accumulator referenced by multimodal message parts.
/// * `audios` - The ordered audio payload accumulator referenced by multimodal message parts.
pub(crate) fn map_generate_message(
    message: &ConversationMessage,
    role_strategy: RoleStrategy,
    images: &mut Vec<image::DynamicImage>,
    audios: &mut Vec<MistralAudioInput>,
) -> Result<Vec<IndexMap<String, MessageContent>>> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    let mut image_parts = Vec::new();
    let mut audio_parts = Vec::new();

    for part in &message.parts {
        match part {
            ContentPart::Text(text) => text_parts.push(text.clone()),
            ContentPart::ToolCall(call) => tool_calls.push(call.clone()),
            ContentPart::ToolResult(result) => tool_results.push(result.clone()),
            ContentPart::Image(image) => image_parts.push(image),
            ContentPart::Video(_) => {
                return Err(Error::UnsupportedMessagePart {
                    part: "video input",
                });
            }
            ContentPart::Audio(audio) => audio_parts.push(audio),
        }
    }

    if !tool_results.is_empty() {
        if !text_parts.is_empty()
            || !tool_calls.is_empty()
            || !image_parts.is_empty()
            || !audio_parts.is_empty()
            || message.role != MessageRole::Tool
        {
            return Err(Error::UnsupportedMessagePart {
                part: "mixed tool-result message",
            });
        }

        return tool_results
            .iter()
            .map(map_tool_result_message)
            .collect::<Result<Vec<_>>>();
    }

    if !tool_calls.is_empty() && (!image_parts.is_empty() || !audio_parts.is_empty()) {
        return Err(Error::UnsupportedMessagePart {
            part: "mixed tool-call and multimodal message",
        });
    }

    let mut mapped = IndexMap::new();
    mapped.insert(
        "role".to_string(),
        Either::Left(map_role(message.role, role_strategy).to_string()),
    );

    if image_parts.is_empty() && audio_parts.is_empty() {
        let content = if !text_parts.is_empty() {
            text_parts.join("\n")
        } else if !tool_calls.is_empty() {
            serialize_tool_calls(&tool_calls)?
        } else {
            String::new()
        };
        mapped.insert("content".to_string(), Either::Left(content));
    } else {
        let mut content_parts = Vec::new();

        for image in image_parts {
            images.push(map_image_input(image)?);
            content_parts.push(IndexMap::from([(
                "type".to_string(),
                serde_json::Value::String("image".to_string()),
            )]));
        }

        for audio in audio_parts {
            audios.push(map_audio_input(audio)?);
            content_parts.push(IndexMap::from([(
                "type".to_string(),
                serde_json::Value::String("audio".to_string()),
            )]));
        }

        content_parts.push(IndexMap::from([
            (
                "type".to_string(),
                serde_json::Value::String("text".to_string()),
            ),
            (
                "text".to_string(),
                serde_json::Value::String(text_parts.join("\n")),
            ),
        ]));
        mapped.insert("content".to_string(), Either::Right(content_parts));
    }

    if !tool_calls.is_empty() {
        mapped.insert("tool_calls".to_string(), map_tool_calls_field(&tool_calls)?);
    }

    Ok(vec![mapped])
}
