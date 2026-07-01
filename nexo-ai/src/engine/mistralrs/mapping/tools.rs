use crate::{Error, Result};
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    Function as MistralFunction, MessageContent, Tool, ToolChoice as MistralToolChoice, ToolType,
};
use nexo_core::{
    MessageRole, RoleStrategy, ToolCall, ToolChoice, ToolDefinition, ToolResult, ToolResultContent,
};
use std::collections::HashMap;

/// Maps shared tool definitions into the Mistral.rs tool schema representation.
///
/// # Arguments
///
/// * `definitions` - The tool definitions exposed to the model for the current request.
pub(crate) fn map_tool_definitions(definitions: &[ToolDefinition]) -> Result<Option<Vec<Tool>>> {
    if definitions.is_empty() {
        return Ok(None);
    }

    definitions
        .iter()
        .map(|definition| {
            let parameters = match &definition.parameters {
                serde_json::Value::Null => None,
                serde_json::Value::Object(object) => Some(
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<HashMap<_, _>>(),
                ),
                other => {
                    return Err(Error::InvalidToolPayload {
                        tool_name: definition.name.clone(),
                        message: format!(
                            "expected an object or null for tool parameters, got {other}"
                        ),
                    });
                }
            };

            Ok(Tool {
                tp: ToolType::Function,
                function: MistralFunction {
                    description: Some(definition.description.clone()),
                    name: definition.name.clone(),
                    parameters,
                    strict: None,
                },
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Maps the shared tool choice policy into the Mistral.rs tool-choice representation.
///
/// # Arguments
///
/// * `choice` - The shared tool-choice policy applied to the request.
/// * `tools` - The mapped Mistral.rs tools available to the model.
pub(crate) fn map_tool_choice(
    choice: &ToolChoice,
    tools: Option<&[Tool]>,
) -> Result<Option<MistralToolChoice>> {
    let Some(tools) = tools else {
        return Ok(None);
    };

    if tools.is_empty() {
        return Ok(None);
    }

    match choice {
        ToolChoice::Disabled => Ok(None),
        ToolChoice::Automatic => Ok(Some(MistralToolChoice::Auto)),
        ToolChoice::Specific { name } => {
            let tool = tools
                .iter()
                .find(|tool| tool.function.name == *name)
                .cloned()
                .ok_or_else(|| Error::InvalidToolPayload {
                    tool_name: name.clone(),
                    message: "forced tool choice was not present in the request tool list"
                        .to_string(),
                })?;
            Ok(Some(MistralToolChoice::Tool(tool)))
        }
    }
}

/// Maps fully formed tool calls into the message field expected by Mistral.rs.
///
/// # Arguments
///
/// * `calls` - The shared tool calls that should be serialized into a message field.
pub(crate) fn map_tool_calls_field(calls: &[ToolCall]) -> Result<MessageContent> {
    let mut mapped_calls = Vec::with_capacity(calls.len());

    for call in calls {
        let mut mapped_call = IndexMap::new();
        mapped_call.insert(
            "id".to_string(),
            serde_json::Value::String(call.id.to_string()),
        );
        mapped_call.insert(
            "type".to_string(),
            serde_json::Value::String("function".to_string()),
        );

        let arguments = call.arguments.clone();
        let mut function = serde_json::Map::new();
        function.insert(
            "name".to_string(),
            serde_json::Value::String(call.name.clone()),
        );
        function.insert("arguments".to_string(), arguments);
        mapped_call.insert("function".to_string(), serde_json::Value::Object(function));
        mapped_calls.push(mapped_call);
    }

    Ok(Either::Right(mapped_calls))
}

/// Serializes tool calls into the plain-text content fallback used by Mistral.rs chat messages.
///
/// # Arguments
///
/// * `calls` - The shared tool calls that should be serialized into JSON text.
pub(crate) fn serialize_tool_calls(calls: &[ToolCall]) -> Result<String> {
    if calls.len() == 1 {
        return Ok(serde_json::to_string(&call_to_value(&calls[0]))?);
    }

    serde_json::to_string(
        &calls
            .iter()
            .map(call_to_value)
            .collect::<Vec<serde_json::Value>>(),
    )
    .map_err(Into::into)
}

/// Maps a tool result message into the message representation expected by Mistral.rs.
///
/// # Arguments
///
/// * `result` - The tool result that should be inserted into the conversation transcript.
pub(crate) fn map_tool_result_message(
    result: &ToolResult,
) -> Result<IndexMap<String, MessageContent>> {
    let mut mapped = IndexMap::new();
    mapped.insert("role".to_string(), Either::Left("tool".to_string()));
    mapped.insert("name".to_string(), Either::Left(result.tool_name.clone()));
    mapped.insert(
        "content".to_string(),
        Either::Left(match &result.content {
            ToolResultContent::Text(text) => text.clone(),
            ToolResultContent::Json(value) => serde_json::to_string(value)?,
        }),
    );
    Ok(mapped)
}

/// Maps a shared message role into the string role expected by Mistral.rs.
///
/// # Arguments
///
/// * `role` - The shared message role to encode.
/// * `strategy` - The role strategy required by the target model.
pub(crate) fn map_role(role: MessageRole, strategy: RoleStrategy) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::Developer => {
            if matches!(strategy, RoleStrategy::MergeDeveloperIntoSystem) {
                "system"
            } else {
                "developer"
            }
        }
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Converts a shared tool call into the JSON value form used by fallback serialization.
///
/// # Arguments
///
/// * `call` - The tool call to serialize into a JSON value.
fn call_to_value(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id.to_string(),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        }
    })
}
