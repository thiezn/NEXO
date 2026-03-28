use serde::{Deserialize, Serialize};

use super::{openai, qwen_cli, whisper};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub llama_url: String,
    pub tts_url: String,
    pub whisper_url: String,
    pub vision_url: String,
    pub qwen_image_cli: String,
    pub image_output_dir: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            llama_url: "http://127.0.0.1:8001".into(),
            tts_url: "http://127.0.0.1:8002".into(),
            whisper_url: "http://127.0.0.1:8003".into(),
            vision_url: "http://127.0.0.1:8004".into(),
            qwen_image_cli: "qwen-image-mps".into(),
            image_output_dir: "/tmp/nexo-images".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub messages: Vec<ChatMessage>,
    /// Tools in OpenAI function-calling format (raw JSON objects).
    pub tools: Vec<serde_json::Value>,
    pub max_tokens: usize,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResponse {
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<String>,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ImageAnalysisRequest {
    pub image_data: Vec<u8>,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageAnalysisResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ListenRequest {
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListenResponse {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct TalkRequest {
    pub text: String,
    pub voice: String,
    pub instruct: Option<String>,
    pub language: Option<String>,
    pub speed: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TalkResponse {
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ImagineRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: u64,
    pub batch_size: u32,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub index: u32,
}

#[derive(Debug, Clone)]
pub struct ImagineResponse {
    pub images: Vec<GeneratedImage>,
    pub seed_used: u64,
    pub inference_time_ms: u64,
}

pub struct InferenceClients {
    http: reqwest::Client,
    config: InferenceConfig,
}

impl InferenceClients {
    pub fn new(config: InferenceConfig) -> Self {
        let http = reqwest::Client::new();
        Self { http, config }
    }

    /// Chat completion via llama-server (Qwen3.5-35B-A3B).
    pub async fn chat(&self, req: ChatRequest) -> anyhow::Result<ChatResponse> {
        openai::chat(&self.http, &self.config.llama_url, req).await
    }

    /// Tool-augmented chat completion via llama-server.
    pub async fn tool_call(&self, req: ToolCallRequest) -> anyhow::Result<ToolCallResponse> {
        openai::tool_call(&self.http, &self.config.llama_url, req).await
    }

    /// Multimodal image analysis via vllm-mlx (Qwen3.5-9B).
    pub async fn analyze_image(
        &self,
        req: ImageAnalysisRequest,
    ) -> anyhow::Result<ImageAnalysisResponse> {
        openai::analyze_image(&self.http, &self.config.vision_url, req).await
    }

    /// Text-to-speech via mlx-tts-server (Qwen3-TTS).
    pub async fn talk(&self, req: TalkRequest) -> anyhow::Result<TalkResponse> {
        openai::talk(&self.http, &self.config.tts_url, req).await
    }

    /// Speech-to-text via whisper-server (Whisper large-v3-turbo).
    pub async fn listen(&self, req: ListenRequest) -> anyhow::Result<ListenResponse> {
        whisper::transcribe(&self.http, &self.config.whisper_url, req).await
    }

    /// Image generation via qwen-image-mps CLI subprocess.
    pub async fn imagine(&self, req: ImagineRequest) -> anyhow::Result<ImagineResponse> {
        qwen_cli::generate(&self.config.qwen_image_cli, &self.config.image_output_dir, req).await
    }
}
