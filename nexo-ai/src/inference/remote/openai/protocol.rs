use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

/// Metadata for a model reported by an OpenAI-compatible `/v1/models` endpoint.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct OpenAiModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiModelsResponse {
    pub data: Vec<OpenAiModelInfo>,
}

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiSpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruct: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    pub response_format: String,
    pub stream: bool,
}

/// OpenAI-compatible audio transcription request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiTranscriptionRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub verbose: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiToolDefinition>>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: Option<OpenAiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiRequestToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiRequestToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiRequestToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiRequestToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlDetail },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: AudioDetail },
}

#[derive(Debug, Serialize)]
pub struct ImageUrlDetail {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetailLevel>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetailLevel {
    Auto,
    Low,
    High,
}

#[derive(Debug, Serialize)]
pub struct AudioDetail {
    /// Base64-encoded audio data.
    pub data: String,

    /// Audio format, e.g. "wav" or "mp3".
    pub format: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAiToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDefinition,
}

#[derive(Debug, Serialize)]
pub struct OpenAiFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl From<&nexo_spec::tool::ToolSpec> for OpenAiToolDefinition {
    fn from(tool: &nexo_spec::tool::ToolSpec) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: OpenAiFunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        }
    }
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Deserialize)]
pub struct OpenAiChatResponse {
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseMessage {
    #[serde(default, deserialize_with = "deserialize_response_content")]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default, deserialize_with = "deserialize_tool_calls")]
    pub tool_calls: Vec<OpenAiResponseToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseToolCall {
    pub function: OpenAiResponseToolFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiUsage {
    pub completion_tokens: Option<usize>,
}

fn deserialize_response_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text)),
        Some(serde_json::Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                .collect::<String>();
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        }
        Some(other) => Err(D::Error::custom(format!(
            "unsupported chat response content: {other}"
        ))),
    }
}

fn deserialize_tool_calls<'de, D>(deserializer: D) -> Result<Vec<OpenAiResponseToolCall>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Vec<OpenAiResponseToolCall>>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn text_content_serializes_as_string() {
        let msg = OpenAiMessage {
            role: "user".to_string(),
            content: Some(OpenAiContent::Text("hello".to_string())),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn parts_content_serializes_as_array() {
        let msg = OpenAiMessage {
            role: "user".to_string(),
            content: Some(OpenAiContent::Parts(vec![
                OpenAiContentPart::Text {
                    text: "describe this".to_string(),
                },
                OpenAiContentPart::ImageUrl {
                    image_url: ImageUrlDetail {
                        url: "data:image/png;base64,abc".to_string(),
                        detail: None,
                    },
                },
            ])),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        let parts = json["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
    }

    #[test]
    fn audio_content_part_serializes() {
        let part = OpenAiContentPart::InputAudio {
            input_audio: AudioDetail {
                data: "AAAA".to_string(),
                format: "wav".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "input_audio");
        assert_eq!(json["input_audio"]["format"], "wav");
    }

    #[test]
    fn chat_request_serializes() {
        let req = OpenAiChatRequest {
            model: "test-model".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(OpenAiContent::Text("hi".to_string())),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: None,
            tools: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "test-model");
        assert!(json.get("top_p").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn tools_serialize_in_openai_function_shape() {
        let req = OpenAiChatRequest {
            model: "test-model".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(OpenAiContent::Text("hi".to_string())),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: Some(0.9),
            tools: Some(vec![OpenAiToolDefinition::from(
                &nexo_spec::tool::ToolSpec {
                    name: "get_weather".into(),
                    description: "Get the weather".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }),
                    ..Default::default()
                },
            )]),
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            json["tools"][0]["function"]["description"],
            "Get the weather"
        );
    }

    #[test]
    fn assistant_tool_turn_can_serialize_null_content() {
        let msg = OpenAiMessage {
            role: "assistant".to_string(),
            content: None,
            tool_call_id: None,
            name: None,
            tool_calls: Some(vec![OpenAiRequestToolCall {
                id: "call-1".to_string(),
                tool_type: "function".to_string(),
                function: OpenAiRequestToolFunction {
                    name: "io.bash".to_string(),
                    arguments: r#"{"command":"ls"}"#.to_string(),
                },
            }]),
        };

        let json = serde_json::to_value(&msg).unwrap();
        assert!(json["content"].is_null());
        assert_eq!(json["tool_calls"][0]["id"], "call-1");
    }

    #[test]
    fn chat_response_deserializes() {
        let json = r#"{
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"completion_tokens": 2}
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("hello world")
        );
        assert_eq!(resp.usage.unwrap().completion_tokens, Some(2));
    }

    #[test]
    fn chat_response_deserializes_tool_calls() {
        let json = r#"{
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Amsterdam\"}"
                    }
                }]
            }}],
            "usage": {"completion_tokens": 8}
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.tool_calls.len(), 1);
        assert_eq!(
            resp.choices[0].message.tool_calls[0].function.name,
            "get_weather"
        );
        assert_eq!(
            resp.choices[0].message.tool_calls[0].function.arguments,
            r#"{"city":"Amsterdam"}"#
        );
    }

    #[test]
    fn chat_response_deserializes_null_tool_calls() {
        let json = r#"{
            "choices": [{"message": {
                "role": "assistant",
                "content": "hello",
                "reasoning": null,
                "tool_calls": null,
                "tool_call_id": null,
                "name": null
            }}],
            "usage": null
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("hello"));
        assert!(resp.choices[0].message.tool_calls.is_empty());
        assert_eq!(resp.choices[0].message.reasoning, None);
    }

    #[test]
    fn chat_response_deserializes_output_text_parts() {
        let json = r#"{
            "choices": [{"message": {
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"}
                ],
                "reasoning": "chain",
                "tool_calls": []
            }}]
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("hello world")
        );
        assert_eq!(resp.choices[0].message.reasoning.as_deref(), Some("chain"));
    }
}
