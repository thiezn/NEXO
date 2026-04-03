use candle_core::Tensor;
use serde::{Deserialize, Serialize};

pub use nexo_spec::model::ModelCategory;

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// Role within a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Request for a chat completion.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub session_id: Option<String>,
}

/// Response from a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

/// Request for tool-augmented generation.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<nexo_spec::tool::ToolSpec>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub session_id: Option<String>,
}

/// A single tool invocation produced by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Response containing zero or more tool calls.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResponse {
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<String>,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Vision / Image analysis
// ---------------------------------------------------------------------------

/// Request to analyse an image.
#[derive(Debug, Clone)]
pub struct ImageAnalysisRequest {
    pub image_data: Vec<u8>,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
}

/// Response from image analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ImageAnalysisResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Listen (speech-to-text)
// ---------------------------------------------------------------------------

/// Request to transcribe audio.
#[derive(Debug, Clone)]
pub struct ListenRequest {
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub language: Option<String>,
}

/// A time-aligned transcription segment.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Response from speech-to-text transcription.
#[derive(Debug, Clone, Serialize)]
pub struct ListenResponse {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Talk (text-to-speech)
// ---------------------------------------------------------------------------

/// Request to synthesise speech.
#[derive(Debug, Clone)]
pub struct TalkRequest {
    pub text: String,
    pub voice_description: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: u64,
}

/// Response containing synthesised audio.
#[derive(Debug, Clone)]
pub struct TalkResponse {
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Imagine (text-to-image)
// ---------------------------------------------------------------------------

/// Request to generate images from a prompt.
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

/// A single generated image.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub index: u32,
}

/// Response containing one or more generated images.
#[derive(Debug, Clone)]
pub struct ImagineResponse {
    pub images: Vec<GeneratedImage>,
    pub seed_used: u64,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Embed (text-to-embedding)
// ---------------------------------------------------------------------------

/// Request to generate text embeddings.
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

/// Response containing embedding vectors.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimensions: usize,
    pub inference_time_ms: u64,
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Snapshot of a single layer's K/V cache state, used for save/restore.
#[derive(Debug, Clone)]
pub struct LayerKvSnapshot {
    pub layer_idx: usize,
    pub is_sliding: bool,
    pub k_data: Option<Tensor>,
    pub v_data: Option<Tensor>,
    pub offset: usize,
    pub current_seq_len: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // -- ChatRole serde --

    #[test]
    fn chat_role_serde_roundtrip() {
        for role in [ChatRole::System, ChatRole::User, ChatRole::Assistant] {
            let json = serde_json::to_string(&role).unwrap();
            let parsed: ChatRole = serde_json::from_str(&json).unwrap();
            assert_eq!(role, parsed);
        }
    }

    // -- ChatMessage serde --

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, ChatRole::User);
        assert_eq!(parsed.content, "hello");
    }

    // -- ToolCall serde --

    #[test]
    fn tool_call_serde_roundtrip() {
        let tc = ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Amsterdam"}),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let parsed: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "get_weather");
        assert_eq!(parsed.arguments["city"], "Amsterdam");
    }

    // -- ChatResponse serialization --

    #[test]
    fn chat_response_serializes() {
        let resp = ChatResponse {
            text: "hi".into(),
            tokens_generated: 1,
            inference_time_ms: 42,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["text"], "hi");
        assert_eq!(v["tokens_generated"], 1);
        assert_eq!(v["inference_time_ms"], 42);
    }

    // -- ToolCallResponse serialization --

    #[test]
    fn tool_call_response_serializes() {
        let resp = ToolCallResponse {
            tool_calls: vec![ToolCall {
                name: "search".into(),
                arguments: serde_json::json!({}),
            }],
            reasoning: Some("because".into()),
            tokens_generated: 10,
            inference_time_ms: 100,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["tool_calls"][0]["name"], "search");
        assert_eq!(v["reasoning"], "because");
    }

    // -- ImageAnalysisResponse serialization --

    #[test]
    fn image_analysis_response_serializes() {
        let resp = ImageAnalysisResponse {
            text: "a cat".into(),
            tokens_generated: 2,
            inference_time_ms: 50,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("a cat"));
    }

    // -- ListenResponse serialization --

    #[test]
    fn listen_response_serializes() {
        let resp = ListenResponse {
            text: "hello world".into(),
            segments: vec![TranscriptionSegment {
                text: "hello world".into(),
                start_ms: 0,
                end_ms: 1000,
            }],
            language: Some("en".into()),
            inference_time_ms: 200,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["segments"][0]["start_ms"], 0);
        assert_eq!(v["segments"][0]["end_ms"], 1000);
        assert_eq!(v["language"], "en");
    }

    // -- TranscriptionSegment serialization --

    #[test]
    fn transcription_segment_serializes() {
        let seg = TranscriptionSegment {
            text: "word".into(),
            start_ms: 100,
            end_ms: 200,
        };
        let json = serde_json::to_string(&seg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["text"], "word");
        assert_eq!(v["start_ms"], 100);
        assert_eq!(v["end_ms"], 200);
    }

    // -- Default-like construction tests --

    #[test]
    fn chat_request_can_be_constructed() {
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::System,
                content: "You are helpful.".into(),
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn imagine_request_can_be_constructed() {
        let req = ImagineRequest {
            prompt: "a sunset".into(),
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5,
            seed: 42,
            batch_size: 1,
        };
        assert_eq!(req.width, 512);
        assert_eq!(req.batch_size, 1);
    }

    #[test]
    fn talk_request_can_be_constructed() {
        let req = TalkRequest {
            text: "hello".into(),
            voice_description: "warm female".into(),
            max_tokens: 500,
            temperature: 0.6,
            seed: 123,
        };
        assert_eq!(req.text, "hello");
    }

    #[test]
    fn listen_request_can_be_constructed() {
        let req = ListenRequest {
            pcm_samples: vec![0.0; 16000],
            sample_rate: 16000,
            language: None,
        };
        assert_eq!(req.pcm_samples.len(), 16000);
        assert!(req.language.is_none());
    }

    #[test]
    fn generated_image_can_be_constructed() {
        let img = GeneratedImage {
            data: vec![0u8; 100],
            width: 10,
            height: 10,
            index: 0,
        };
        assert_eq!(img.data.len(), 100);
    }

    #[test]
    fn imagine_response_can_be_constructed() {
        let resp = ImagineResponse {
            images: vec![],
            seed_used: 42,
            inference_time_ms: 5000,
        };
        assert!(resp.images.is_empty());
        assert_eq!(resp.seed_used, 42);
    }

    #[test]
    fn embed_request_can_be_constructed() {
        let req = EmbedRequest {
            texts: vec!["hello".into(), "world".into()],
        };
        assert_eq!(req.texts.len(), 2);
    }

    #[test]
    fn embed_response_can_be_constructed() {
        let resp = EmbedResponse {
            embeddings: vec![vec![0.1, 0.2, 0.3]],
            dimensions: 3,
            inference_time_ms: 10,
        };
        assert_eq!(resp.embeddings.len(), 1);
        assert_eq!(resp.dimensions, 3);
    }
}
