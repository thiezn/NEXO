use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::mlx_server::MlxModelInfo;

// ── Wire types ───────────────────────────────────────────────────────────────

/// OpenAI-compatible chat completion request.
#[derive(Debug, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: OpenAiContent,
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
}

#[derive(Debug, Serialize)]
pub struct AudioDetail {
    pub data: String,
    pub format: String,
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

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseMessage {
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiUsage {
    pub completion_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<MlxModelInfo>,
}

// ── Client ───────────────────────────────────────────────────────────────────

/// Async HTTP client for an OpenAI-compatible API.
///
/// By default points to a locally running mlx_vlm server, but can target any
/// OpenAI-compatible endpoint.
pub struct OpenAiClient {
    client: reqwest::Client,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Send a chat completion request.
    pub async fn chat_completion(&self, request: &OpenAiChatRequest) -> Result<OpenAiChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(request)
            .send()
            .await
            .context("failed to POST /v1/chat/completions")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("/v1/chat/completions returned {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse chat completion response")
    }

    /// List models via `GET /v1/models`.
    pub async fn list_models(&self) -> Result<Vec<MlxModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to GET /v1/models")?;

        if !resp.status().is_success() {
            bail!("/v1/models returned status {}", resp.status());
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .context("failed to parse /v1/models response")?;
        Ok(body.data)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn text_content_serializes_as_string() {
        let msg = OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Text("hello".to_string()),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn parts_content_serializes_as_array() {
        let msg = OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Parts(vec![
                OpenAiContentPart::Text {
                    text: "describe this".to_string(),
                },
                OpenAiContentPart::ImageUrl {
                    image_url: ImageUrlDetail {
                        url: "data:image/png;base64,abc".to_string(),
                    },
                },
            ]),
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
                content: OpenAiContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "test-model");
        assert!(json.get("top_p").is_none());
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
}
