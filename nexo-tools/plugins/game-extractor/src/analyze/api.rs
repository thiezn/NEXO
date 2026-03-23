use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use super::prompt;

pub struct LmClient {
    client: reqwest::Client,
    endpoint: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: Vec<ContentPart>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlData },
}

#[derive(Serialize)]
struct ImageUrlData {
    url: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

impl LmClient {
    pub fn new(endpoint: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    pub async fn describe_image(&self, image_path: &Path) -> Result<String> {
        let bytes = tokio::fs::read(image_path)
            .await
            .with_context(|| format!("Failed to read {}", image_path.display()))?;

        let mime = match image_path.extension().and_then(|e| e.to_str()) {
            Some("png") => "image/png",
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };

        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let data_uri = format!("data:{mime};base64,{encoded}");

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: vec![ContentPart::Text {
                        text: prompt::system_prompt().to_string(),
                    }],
                },
                Message {
                    role: "user".to_string(),
                    content: vec![
                        ContentPart::ImageUrl {
                            image_url: ImageUrlData { url: data_uri },
                        },
                        ContentPart::Text {
                            text: prompt::user_prompt().to_string(),
                        },
                    ],
                },
            ],
            max_tokens: 512,
        };

        let url = format!("{}/v1/chat/completions", self.endpoint);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .with_context(|| format!("Failed to connect to LM Studio at {url}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LM Studio returned {status}: {body}");
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("Failed to parse LM Studio response")?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .context("LM Studio returned no choices")
    }
}
