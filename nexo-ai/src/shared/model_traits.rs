use super::types::*;
use anyhow::Result;

/// KV cache save/restore for session-aware inference.
#[cfg(feature = "candle")]
pub trait KvCacheable {
    /// Current number of cached tokens.
    fn kv_cache_seq_len(&self) -> usize;

    /// Extract K/V tensors from all layers.
    fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>>;

    /// Restore K/V cache from saved snapshots.
    fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()>;

    /// Clear all KV cache state.
    fn clear_kv_cache(&mut self);

    /// Token IDs processed so far in the current session.
    fn processed_tokens(&self) -> &[u32];

    /// Current session ID tracked by the model.
    fn current_session_id(&self) -> Option<&str>;

    /// Set session state (session ID + processed tokens).
    fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>);

    /// Device the model is running on (needed for disk cache restoration).
    fn device(&self) -> &candle_core::Device;

    /// Dtype used by the model (needed for disk cache restoration).
    fn dtype(&self) -> candle_core::DType;
}

/// Common metadata and lifecycle for any model.
pub trait ModelInfo: Send {
    /// Human-readable model name (e.g. "Qwen3-4B").
    fn name(&self) -> &str;

    /// Model family identifier (e.g. "qwen3", "whisper", "flux").
    fn family(&self) -> &str;

    /// The categories this model can serve.
    fn categories(&self) -> &[ModelCategory];

    /// Estimated memory required when loaded, in bytes.
    fn memory_estimate_bytes(&self) -> u64;

    /// Whether the model weights are currently loaded.
    fn is_loaded(&self) -> bool;

    /// Load model weights into memory.
    fn load(&mut self) -> Result<()>;

    /// Unload model weights, freeing memory.
    fn unload(&mut self);

    // ── Category downcasts ─────────────────────────────────────────
    // Override when the model supports a category. Default returns None.

    fn as_chat(&mut self) -> Option<&mut dyn ChatModel> {
        None
    }
    fn as_tool(&mut self) -> Option<&mut dyn ToolModel> {
        None
    }
    fn as_image(&mut self) -> Option<&mut dyn ImageModel> {
        None
    }
    fn as_listen(&mut self) -> Option<&mut dyn ListenModel> {
        None
    }
    fn as_audio_analysis(&mut self) -> Option<&mut dyn AudioAnalysisModel> {
        None
    }
    fn as_talk(&mut self) -> Option<&mut dyn TalkModel> {
        None
    }
    fn as_imagine(&mut self) -> Option<&mut dyn ImagineModel> {
        None
    }
    fn as_embed(&mut self) -> Option<&mut dyn EmbedModel> {
        None
    }
    fn as_multimodal(&mut self) -> Option<&mut dyn MultiModalModel> {
        None
    }
    #[cfg(feature = "candle")]
    fn as_kv_cacheable(&mut self) -> Option<&mut dyn KvCacheable> {
        None
    }
}

/// Text chat completion.
pub trait ChatModel: ModelInfo {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse>;
}

/// Tool-augmented generation.
pub trait ToolModel: ModelInfo {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse>;
}

/// Image understanding / analysis.
pub trait ImageModel: ModelInfo {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse>;
}

/// Speech-to-text transcription.
pub trait ListenModel: ModelInfo {
    fn transcribe(&mut self, request: &ListenRequest) -> Result<ListenResponse>;
}

/// Prompted audio understanding (describe, answer questions about audio).
pub trait AudioAnalysisModel: ModelInfo {
    fn analyze_audio(&mut self, request: &AudioAnalysisRequest) -> Result<AudioAnalysisResponse>;
}

/// Text-to-speech synthesis.
pub trait TalkModel: ModelInfo {
    fn synthesize(&mut self, request: &TalkRequest) -> Result<TalkResponse>;
}

/// Text-to-image generation.
pub trait ImagineModel: ModelInfo {
    fn imagine(&mut self, request: &ImagineRequest) -> Result<ImagineResponse>;
}

/// Text embedding.
pub trait EmbedModel: ModelInfo {
    fn embed(&mut self, request: &EmbedRequest) -> Result<EmbedResponse>;
}

/// Combined text + image + audio inference.
pub trait MultiModalModel: ModelInfo {
    fn multimodal(&mut self, request: &MultiModalRequest) -> Result<MultiModalResponse>;
}
