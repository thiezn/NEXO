use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

use crate::api::model_traits::*;
use crate::api::types::*;
use crate::models::gemma4::common::template::parse_native_tool_calls;

use super::client::OpenAiClient;
use super::protocol::{
    AudioDetail, ImageUrlDetail, OpenAiChatRequest, OpenAiContent, OpenAiContentPart,
    OpenAiMessage, OpenAiRequestToolCall, OpenAiRequestToolFunction, OpenAiResponseMessage,
    OpenAiResponseToolCall, OpenAiToolDefinition,
};

pub trait OpenAiServerControl: Clone + Send + Sync + 'static {
    fn ensure_running(&self) -> Result<()>;
    fn unload_model(&self, model_id: &str) -> Result<()>;
}

impl OpenAiServerControl for () {
    fn ensure_running(&self) -> Result<()> {
        Ok(())
    }

    fn unload_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }
}

pub trait OpenAiFamilyAdapter: Clone + Send + Sync + 'static {
    fn family(&self) -> &'static str;

    fn resolve_request_model_id(
        &self,
        _model_name: &str,
        model_dir: &Path,
        explicit: Option<&str>,
    ) -> String {
        explicit
            .map(str::to_string)
            .unwrap_or_else(|| model_dir.to_string_lossy().to_string())
    }

    fn parse_tool_response(
        &self,
        message: &OpenAiResponseMessage,
    ) -> (Vec<ToolCall>, Option<String>) {
        let tool_calls = parse_wire_tool_calls(&message.tool_calls);
        let reasoning = message
            .reasoning
            .clone()
            .or_else(|| message.content.clone())
            .filter(|text| !text.trim().is_empty());
        (tool_calls, reasoning)
    }
}

pub fn parse_wire_tool_calls(tool_calls: &[OpenAiResponseToolCall]) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .filter_map(|call| {
            serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .map(|arguments| ToolCall {
                    name: call.function.name.clone(),
                    arguments,
                })
                .map_err(|error| {
                    tracing::warn!(
                        "failed to parse OpenAI tool arguments for '{}': {error}",
                        call.function.name
                    )
                })
                .ok()
        })
        .collect()
}

/// Generic model adapter for OpenAI-compatible inference backends.
pub struct OpenAiModel<F, S = ()> {
    name: String,
    model_dir: PathBuf,
    request_model_id: Option<String>,
    memory_bytes: u64,
    categories: Vec<ModelCategory>,
    family: F,
    server: S,
    client: OpenAiClient,
    loaded: bool,
}

impl<F, S> OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
    pub fn new(
        name: impl Into<String>,
        model_dir: PathBuf,
        memory_bytes: u64,
        categories: Vec<ModelCategory>,
        family: F,
        server: S,
        base_url: &str,
    ) -> Self {
        Self {
            name: name.into(),
            model_dir,
            request_model_id: None,
            memory_bytes,
            categories,
            family,
            server,
            client: OpenAiClient::new(base_url),
            loaded: false,
        }
    }

    pub fn with_request_model_id(mut self, request_model_id: impl Into<String>) -> Self {
        self.request_model_id = Some(request_model_id.into());
        self
    }

    fn model_id(&self) -> String {
        self.family.resolve_request_model_id(
            &self.name,
            &self.model_dir,
            self.request_model_id.as_deref(),
        )
    }

    fn ensure_loaded(&self) -> Result<()> {
        if !self.loaded {
            bail!("model '{}' not loaded", self.name);
        }
        Ok(())
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
    }

    fn complete_message(
        &mut self,
        oai_req: &OpenAiChatRequest,
    ) -> Result<(OpenAiResponseMessage, usize, u64)> {
        self.ensure_loaded()?;
        let start = Instant::now();
        let resp = Self::block_on(self.client.chat_completion(oai_req))?;
        let message = resp
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .unwrap_or(OpenAiResponseMessage {
                content: None,
                reasoning: None,
                tool_calls: Vec::new(),
            });
        let tokens = resp
            .usage
            .and_then(|usage| usage.completion_tokens)
            .unwrap_or(0);
        Ok((message, tokens, start.elapsed().as_millis() as u64))
    }

    fn complete(&mut self, oai_req: &OpenAiChatRequest) -> Result<(String, usize, u64)> {
        let (message, tokens, elapsed_ms) = self.complete_message(oai_req)?;
        Ok((message.content.unwrap_or_default(), tokens, elapsed_ms))
    }
}

impl<F, S> ModelInfo for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        self.family.family()
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
        self.server.ensure_running()?;
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        if !self.loaded {
            return;
        }
        let _ = self.server.unload_model(&self.model_id());
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
        self.categories
            .contains(&ModelCategory::Listen)
            .then_some(self as &mut dyn AudioAnalysisModel)
    }

    fn as_multimodal(&mut self) -> Option<&mut dyn MultiModalModel> {
        Some(self)
    }
}

impl<F, S> ChatModel for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
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

impl<F, S> ToolModel for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> {
        let oai_req = build_tool_request(&self.model_id(), request);
        let (message, tokens_generated, inference_time_ms) = self.complete_message(&oai_req)?;
        let (tool_calls, reasoning) = self.family.parse_tool_response(&message);

        Ok(ToolCallResponse {
            tool_calls,
            reasoning,
            tokens_generated,
            inference_time_ms,
        })
    }
}

impl<F, S> ImageModel for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
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

impl<F, S> AudioAnalysisModel for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
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

impl<F, S> MultiModalModel for OpenAiModel<F, S>
where
    F: OpenAiFamilyAdapter,
    S: OpenAiServerControl,
{
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

fn chat_messages_to_openai(messages: &[ChatMessage]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let tool_calls = assistant_tool_calls_for_history(messages, index, message);
            OpenAiMessage {
                role: match message.role {
                    ChatRole::System => "system".to_string(),
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                    ChatRole::Tool => "tool".to_string(),
                },
                content: assistant_content_for_history(message, tool_calls.as_deref()),
                tool_call_id: message.tool_call_id.clone(),
                name: message.tool_name.clone(),
                tool_calls,
            }
        })
        .collect()
}

fn assistant_content_for_history(
    message: &ChatMessage,
    tool_calls: Option<&[OpenAiRequestToolCall]>,
) -> Option<OpenAiContent> {
    if tool_calls.is_some() {
        let visible_text = strip_native_tool_call_blocks(&message.content);
        if visible_text.is_empty() {
            return None;
        }
        return Some(OpenAiContent::Text(visible_text));
    }

    Some(OpenAiContent::Text(message.content.clone()))
}

fn assistant_tool_calls_for_history(
    messages: &[ChatMessage],
    index: usize,
    message: &ChatMessage,
) -> Option<Vec<OpenAiRequestToolCall>> {
    if message.role != ChatRole::Assistant {
        return None;
    }

    let tool_calls = parse_native_tool_calls(&message.content);
    if tool_calls.is_empty() {
        return None;
    }

    let following_tool_messages: Vec<_> = messages[index + 1..]
        .iter()
        .take_while(|next| next.role == ChatRole::Tool)
        .collect();

    Some(
        tool_calls
            .into_iter()
            .enumerate()
            .map(|(offset, tool_call)| OpenAiRequestToolCall {
                id: following_tool_messages
                    .get(offset)
                    .and_then(|next| next.tool_call_id.clone())
                    .unwrap_or_else(|| format!("history-call-{index}-{offset}")),
                tool_type: "function".to_string(),
                function: OpenAiRequestToolFunction {
                    name: tool_call.name,
                    arguments: serde_json::to_string(&tool_call.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            })
            .collect(),
    )
}

fn strip_native_tool_call_blocks(content: &str) -> String {
    let mut visible = String::new();
    let mut rest = content;

    while let Some(start) = rest.find("<|tool_call>call:") {
        visible.push_str(&rest[..start]);
        let after = &rest[start + "<|tool_call>call:".len()..];
        if let Some(end) = after.find("<tool_call|>") {
            rest = &after[end + "<tool_call|>".len()..];
        } else {
            rest = "";
            break;
        }
    }

    visible.push_str(rest);
    visible.trim().to_string()
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
            content: Some(OpenAiContent::Parts(parts)),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }],
        max_tokens,
        temperature,
        top_p: None,
        tools: None,
    }
}

fn build_chat_request(model_id: &str, request: &ChatRequest) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model_id.to_string(),
        messages: chat_messages_to_openai(&request.messages),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: Some(request.top_p),
        tools: None,
    }
}

fn build_tool_request(model_id: &str, request: &ToolCallRequest) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model_id.to_string(),
        messages: chat_messages_to_openai(&request.messages),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: Some(request.top_p),
        tools: Some(
            request
                .tools
                .iter()
                .map(OpenAiToolDefinition::from)
                .collect(),
        ),
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
    let mut media_parts = Vec::with_capacity(media_count);

    for image in &request.images {
        let data_uri = format!(
            "data:{};base64,{}",
            image.mime_type,
            B64.encode(&image.data)
        );
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
        && let Some(last_user) = messages.iter_mut().rfind(|message| message.role == "user")
    {
        let text = match &last_user.content {
            Some(OpenAiContent::Text(text)) => text.clone(),
            Some(OpenAiContent::Parts(_)) | None => String::new(),
        };
        let mut parts = media_parts;
        if !text.is_empty() {
            parts.push(OpenAiContentPart::Text { text });
        }
        last_user.content = Some(OpenAiContent::Parts(parts));
    }

    OpenAiChatRequest {
        model: model_id.to_string(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: Some(request.top_p),
        tools: None,
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

fn encode_pcm_to_wav_base64(samples: &[f32], sample_rate: u32) -> String {
    let buf = crate::audio::AudioBuffer::new(samples.to_vec(), sample_rate, 1);
    match crate::audio::encode_wav(&buf) {
        Ok(wav_bytes) => B64.encode(&wav_bytes),
        Err(error) => {
            tracing::warn!("WAV encoding failed, sending empty audio: {error}");
            String::new()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct NoopFamily;

    impl OpenAiFamilyAdapter for NoopFamily {
        fn family(&self) -> &'static str {
            "test"
        }
    }

    #[test]
    fn build_tool_request_serializes_native_tools() {
        let request = ToolCallRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "What is the weather in Amsterdam?".into(),
                tool_call_id: None,
                tool_name: None,
            }],
            tools: vec![nexo_spec::tool::ToolSpec {
                name: "get_weather".into(),
                description: "Get the current weather for a city".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }),
                ..Default::default()
            }],
            max_tokens: 64,
            temperature: 0.1,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };

        let wire = build_tool_request("model-id", &request);
        assert_eq!(wire.messages.len(), 1);
        assert!(wire.tools.is_some());
        assert_eq!(wire.tools.as_ref().unwrap().len(), 1);
        assert_eq!(wire.tools.as_ref().unwrap()[0].tool_type, "function");
        assert_eq!(wire.tools.as_ref().unwrap()[0].function.name, "get_weather");

        let user = match &wire.messages[0].content {
            Some(OpenAiContent::Text(text)) => text,
            Some(OpenAiContent::Parts(_)) => panic!("expected text user content"),
            None => panic!("expected user content"),
        };
        assert_eq!(user, "What is the weather in Amsterdam?");
        assert!(wire.messages[0].tool_call_id.is_none());
        assert!(wire.messages[0].tool_calls.is_none());
    }

    #[test]
    fn chat_messages_to_openai_preserves_tool_metadata() {
        let wire = chat_messages_to_openai(&[ChatMessage::with_tool_metadata(
            ChatRole::Tool,
            "stdout: hello",
            Some("call-1".into()),
            Some("io.bash".into()),
        )]);

        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "tool");
        assert_eq!(wire[0].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(wire[0].name.as_deref(), Some("io.bash"));
        assert!(wire[0].tool_calls.is_none());
    }

    #[test]
    fn chat_messages_to_openai_reconstructs_assistant_tool_calls() {
        let wire = chat_messages_to_openai(&[
            ChatMessage::new(
                ChatRole::Assistant,
                concat!(
                    "<|tool_call>call:io.bash{command:<|\"|>ls -ltrah<|\"|>}<tool_call|>",
                    "<|tool_call>call:files.count{kind:<|\"|>dir<|\"|>}<tool_call|>"
                ),
            ),
            ChatMessage::with_tool_metadata(
                ChatRole::Tool,
                "stdout: first",
                Some("call-1".into()),
                Some("io.bash".into()),
            ),
            ChatMessage::with_tool_metadata(
                ChatRole::Tool,
                "stdout: second",
                Some("call-2".into()),
                Some("files.count".into()),
            ),
        ]);

        assert_eq!(wire.len(), 3);
        let assistant_tool_calls = wire[0].tool_calls.as_ref().unwrap();
        assert_eq!(assistant_tool_calls.len(), 2);
        assert_eq!(assistant_tool_calls[0].id, "call-1");
        assert_eq!(assistant_tool_calls[0].tool_type, "function");
        assert_eq!(assistant_tool_calls[0].function.name, "io.bash");
        assert_eq!(
            assistant_tool_calls[0].function.arguments,
            r#"{"command":"ls -ltrah"}"#
        );
        assert_eq!(assistant_tool_calls[1].id, "call-2");
        assert_eq!(assistant_tool_calls[1].function.name, "files.count");
        assert_eq!(
            assistant_tool_calls[1].function.arguments,
            r#"{"kind":"dir"}"#
        );
        assert!(wire[0].content.is_none());
    }

    #[test]
    fn chat_messages_to_openai_preserves_visible_assistant_text() {
        let wire = chat_messages_to_openai(&[ChatMessage::new(
            ChatRole::Assistant,
            "I will inspect that.<|tool_call>call:io.bash{command:<|\"|>ls<|\"|>}<tool_call|>",
        )]);

        let content = match &wire[0].content {
            Some(OpenAiContent::Text(text)) => text,
            _ => panic!("expected visible assistant text"),
        };
        assert_eq!(content, "I will inspect that.");
        assert!(wire[0].tool_calls.is_some());
    }

    #[test]
    fn default_family_uses_model_dir_as_request_target() {
        let model = OpenAiModel::new(
            "test-model",
            PathBuf::from("/tmp/test-model"),
            1,
            vec![ModelCategory::Chat],
            NoopFamily,
            (),
            "http://127.0.0.1:1234",
        );

        assert_eq!(model.model_id(), "/tmp/test-model");
    }

    #[test]
    fn explicit_request_model_id_overrides_family_default() {
        let model = OpenAiModel::new(
            "test-model",
            PathBuf::from("/tmp/test-model"),
            1,
            vec![ModelCategory::Chat],
            NoopFamily,
            (),
            "http://127.0.0.1:1234",
        )
        .with_request_model_id("mlx-community/gemma-4-E2B-it-4bit");

        assert_eq!(model.model_id(), "mlx-community/gemma-4-E2B-it-4bit");
    }
}
