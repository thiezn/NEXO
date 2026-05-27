use anyhow::{Context, Result, bail};
use reqwest::multipart::{Form, Part};

use super::protocol::{
    OpenAiChatRequest, OpenAiChatResponse, OpenAiModelInfo, OpenAiModelsResponse,
    OpenAiSpeechRequest, OpenAiTranscriptionRequest,
};

/// Async HTTP client for an OpenAI-compatible API.
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

    /// Synthesize speech via `POST /v1/audio/speech`.
    pub async fn synthesize_speech(&self, request: &OpenAiSpeechRequest) -> Result<Vec<u8>> {
        let url = format!("{}/v1/audio/speech", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(request)
            .send()
            .await
            .context("failed to POST /v1/audio/speech")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("/v1/audio/speech returned {status}: {body}");
        }

        Ok(resp
            .bytes()
            .await
            .context("failed to read speech synthesis response body")?
            .to_vec())
    }

    /// Transcribe audio via `POST /v1/audio/transcriptions`.
    pub async fn transcribe_audio(
        &self,
        request: &OpenAiTranscriptionRequest,
        audio_bytes: &[u8],
        file_name: &str,
        mime_type: &str,
    ) -> Result<Vec<u8>> {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let file_part = Part::bytes(audio_bytes.to_vec())
            .file_name(file_name.to_string())
            .mime_str(mime_type)
            .context("failed to set transcription upload MIME type")?;

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", request.model.clone())
            .text("verbose", request.verbose.to_string())
            .text("stream", request.stream.to_string());

        if let Some(language) = &request.language {
            form = form.text("language", language.clone());
        }
        if let Some(max_tokens) = request.max_tokens {
            form = form.text("max_tokens", max_tokens.to_string());
        }
        if let Some(context) = &request.context {
            form = form.text("context", context.clone());
        }
        if let Some(text) = &request.text {
            form = form.text("text", text.clone());
        }

        let resp = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .context("failed to POST /v1/audio/transcriptions")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("/v1/audio/transcriptions returned {status}: {body}");
        }

        Ok(resp
            .bytes()
            .await
            .context("failed to read transcription response body")?
            .to_vec())
    }

    /// List models via `GET /v1/models`.
    pub async fn list_models(&self) -> Result<Vec<OpenAiModelInfo>> {
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

        let body: OpenAiModelsResponse = resp
            .json()
            .await
            .context("failed to parse /v1/models response")?;
        Ok(body.data)
    }
}
