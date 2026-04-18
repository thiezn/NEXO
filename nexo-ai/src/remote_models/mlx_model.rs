use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use tokio::sync::Mutex;

use super::mlx_server::MlxServer;
use super::openai_client::{
    AudioDetail, ImageUrlDetail, OpenAiChatRequest, OpenAiClient, OpenAiContent, OpenAiContentPart,
    OpenAiMessage,
};
use crate::shared::model_traits::*;
use crate::shared::types::*;

/// A model served via the mlx_vlm Python server.
///
/// Implements the nexo-ai model traits by translating requests into
/// OpenAI-compatible API calls. The server auto-loads/caches models
/// when the `model` field in requests changes.
pub struct MlxModel {
    name: String,
    model_dir: PathBuf,
    memory_bytes: u64,
    categories: Vec<ModelCategory>,
    server: Arc<Mutex<MlxServer>>,
    client: OpenAiClient,
    loaded: bool,
}

impl MlxModel {
    pub fn new(
        name: &str,
        _hf_repo: &str,
        model_dir: PathBuf,
        memory_bytes: u64,
        categories: Vec<ModelCategory>,
        server: Arc<Mutex<MlxServer>>,
        base_url: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            model_dir,
            memory_bytes,
            categories,
            server,
            client: OpenAiClient::new(base_url),
            loaded: false,
        }
    }

    /// The model identifier sent in the `model` field of API requests.
    /// Uses the local directory path so the server loads from our downloads.
    fn model_id(&self) -> String {
        self.model_dir.to_string_lossy().to_string()
    }

    fn ensure_loaded(&self) -> Result<()> {
        if !self.loaded {
            bail!("model '{}' not loaded", self.name);
        }
        Ok(())
    }

    /// Bridge async → sync using the current tokio runtime.
    fn block_on<F: std::future::Future<Output = T>, T>(f: F) -> T {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
    }

    /// Send a request and extract the text + token count from the response.
    fn complete(&mut self, oai_req: &OpenAiChatRequest) -> Result<(String, usize, u64)> {
        self.ensure_loaded()?;
        let start = Instant::now();
        let resp = Self::block_on(self.client.chat_completion(oai_req))?;
        let text = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        let tokens = resp.usage.and_then(|u| u.completion_tokens).unwrap_or(0);
        Ok((text, tokens, start.elapsed().as_millis() as u64))
    }
}

// ── ModelInfo ────────────────────────────────────────────────────────────────

impl ModelInfo for MlxModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "mlx"
    }

    fn categories(&self) -> &[ModelCategory] {
        &self.categories
    }

    fn memory_estimate_bytes(&self) -> u64 {
        self.memory_bytes
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        Self::block_on(async { self.server.lock().await.ensure_running().await })?;
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        if !self.loaded {
            return;
        }
        let _ = Self::block_on(async { self.server.lock().await.unload_model().await });
        self.loaded = false;
    }

    fn as_chat(&mut self) -> Option<&mut dyn ChatModel> {
        self.categories
            .contains(&ModelCategory::Chat)
            .then_some(self as &mut dyn ChatModel)
    }

    fn as_tool(&mut self) -> Option<&mut dyn ToolModel> {
        self.categories
            .contains(&ModelCategory::Tool)
            .then_some(self as &mut dyn ToolModel)
    }

    fn as_image(&mut self) -> Option<&mut dyn ImageModel> {
        self.categories
            .contains(&ModelCategory::Image)
            .then_some(self as &mut dyn ImageModel)
    }

    fn as_audio_analysis(&mut self) -> Option<&mut dyn AudioAnalysisModel> {
        None
    }

    fn as_multimodal(&mut self) -> Option<&mut dyn MultiModalModel> {
        Some(self)
    }
}

// ── ChatModel ────────────────────────────────────────────────────────────────

impl ChatModel for MlxModel {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        let oai_req = build_chat_request(&self.model_id(), request);
        let (text, tokens_generated, inference_time_ms) = self.complete(&oai_req)?;
        Ok(ChatResponse {
            text,
            tokens_generated,
            inference_time_ms,
        })
    }
}

// ── ToolModel ────────────────────────────────────────────────────────────────

impl ToolModel for MlxModel {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> {
        // Tool calling via MLX server is text-based
        let oai_req = OpenAiChatRequest {
            model: self.model_id(),
            messages: chat_messages_to_openai(&request.messages),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
        };
        let (raw, tokens_generated, inference_time_ms) = self.complete(&oai_req)?;
        Ok(ToolCallResponse {
            tool_calls: vec![],
            reasoning: Some(raw),
            tokens_generated,
            inference_time_ms,
        })
    }
}

// ── ImageModel ───────────────────────────────────────────────────────────────

impl ImageModel for MlxModel {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        let oai_req = build_image_request(&self.model_id(), request);
        let (text, tokens_generated, inference_time_ms) = self.complete(&oai_req)?;
        Ok(ImageAnalysisResponse {
            text,
            tokens_generated,
            inference_time_ms,
        })
    }
}

// ── AudioAnalysisModel ───────────────────────────────────────────────────────

impl AudioAnalysisModel for MlxModel {
    fn analyze_audio(&mut self, request: &AudioAnalysisRequest) -> Result<AudioAnalysisResponse> {
        let oai_req = build_audio_request(&self.model_id(), request);
        let (text, tokens_generated, inference_time_ms) = self.complete(&oai_req)?;
        Ok(AudioAnalysisResponse {
            text,
            tokens_generated,
            inference_time_ms,
        })
    }
}

// ── MultiModalModel ──────────────────────────────────────────────────────────

impl MultiModalModel for MlxModel {
    fn multimodal(&mut self, request: &MultiModalRequest) -> Result<MultiModalResponse> {
        let oai_req = build_multimodal_request(&self.model_id(), request);
        let (text, tokens_generated, inference_time_ms) = self.complete(&oai_req)?;
        Ok(MultiModalResponse {
            text,
            tokens_generated,
            inference_time_ms,
        })
    }
}

// ── Translation helpers ──────────────────────────────────────────────────────

fn chat_messages_to_openai(messages: &[ChatMessage]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .map(|m| OpenAiMessage {
            role: match m.role {
                ChatRole::System => "system".to_string(),
                ChatRole::User => "user".to_string(),
                ChatRole::Assistant => "assistant".to_string(),
                ChatRole::Tool => "tool".to_string(),
            },
            content: OpenAiContent::Text(m.content.clone()),
        })
        .collect()
}

fn user_parts_request(
    model_id: &str,
    parts: Vec<OpenAiContentPart>,
    max_tokens: usize,
    temperature: f64,
) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model_id.to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Parts(parts),
        }],
        max_tokens,
        temperature,
        top_p: None,
    }
}

fn build_chat_request(model_id: &str, request: &ChatRequest) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model_id.to_string(),
        messages: chat_messages_to_openai(&request.messages),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: Some(request.top_p),
    }
}

fn build_image_request(model_id: &str, request: &ImageAnalysisRequest) -> OpenAiChatRequest {
    let mime = detect_image_mime(&request.image_data);
    let data_uri = format!("data:{mime};base64,{}", B64.encode(&request.image_data));

    user_parts_request(
        model_id,
        vec![
            OpenAiContentPart::ImageUrl {
                image_url: ImageUrlDetail { url: data_uri },
            },
            OpenAiContentPart::Text {
                text: request.prompt.clone(),
            },
        ],
        request.max_tokens,
        request.temperature,
    )
}

fn build_audio_request(model_id: &str, request: &AudioAnalysisRequest) -> OpenAiChatRequest {
    let wav_b64 = encode_pcm_to_wav_base64(&request.pcm_samples, request.sample_rate);

    user_parts_request(
        model_id,
        vec![
            OpenAiContentPart::InputAudio {
                input_audio: AudioDetail {
                    data: wav_b64,
                    format: "wav".to_string(),
                },
            },
            OpenAiContentPart::Text {
                text: request.prompt.clone(),
            },
        ],
        request.max_tokens,
        request.temperature,
    )
}

fn build_multimodal_request(model_id: &str, request: &MultiModalRequest) -> OpenAiChatRequest {
    let mut messages = chat_messages_to_openai(&request.messages);

    let media_count = request.images.len() + usize::from(request.audio.is_some());
    let mut media_parts: Vec<OpenAiContentPart> = Vec::with_capacity(media_count);

    for img in &request.images {
        let data_uri = format!("data:{};base64,{}", img.mime_type, B64.encode(&img.data));
        media_parts.push(OpenAiContentPart::ImageUrl {
            image_url: ImageUrlDetail { url: data_uri },
        });
    }

    if let Some(ref audio) = request.audio {
        let wav_b64 = encode_pcm_to_wav_base64(&audio.pcm_samples, audio.sample_rate);
        media_parts.push(OpenAiContentPart::InputAudio {
            input_audio: AudioDetail {
                data: wav_b64,
                format: "wav".to_string(),
            },
        });
    }

    if !media_parts.is_empty()
        && let Some(last_user) = messages.iter_mut().rfind(|m| m.role == "user")
    {
        let text = match &last_user.content {
            OpenAiContent::Text(t) => t.clone(),
            OpenAiContent::Parts(_) => String::new(),
        };
        let mut parts = media_parts;
        if !text.is_empty() {
            parts.push(OpenAiContentPart::Text { text });
        }
        last_user.content = OpenAiContent::Parts(parts);
    }

    OpenAiChatRequest {
        model: model_id.to_string(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: Some(request.top_p),
    }
}

fn detect_image_mime(data: &[u8]) -> &'static str {
    if data.starts_with(b"\x89PNG") {
        "image/png"
    } else if data.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if data.starts_with(b"GIF8") {
        "image/gif"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

/// Encode PCM f32 samples to WAV bytes (via hound), then base64.
fn encode_pcm_to_wav_base64(samples: &[f32], sample_rate: u32) -> String {
    let buf = crate::audio::AudioBuffer::new(samples.to_vec(), sample_rate, 1);
    match crate::audio::encode_wav(&buf) {
        Ok(wav_bytes) => B64.encode(&wav_bytes),
        Err(e) => {
            tracing::warn!("WAV encoding failed, sending empty audio: {e}");
            String::new()
        }
    }
}
