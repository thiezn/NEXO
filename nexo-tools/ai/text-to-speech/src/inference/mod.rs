pub mod factory;
pub(crate) mod pipelines;

use anyhow::Result;
use local_inference_helpers::progress::ProgressCallback;

/// Controls how model components are loaded during inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Load all components at once, keep hot.
    #[default]
    Eager,
    /// Load-use-drop per component, minimizing peak memory.
    Sequential,
}

/// Request for TTS inference.
#[derive(Debug, Clone)]
pub struct TTSRequest {
    pub text: String,
    pub description: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: u64,
}

/// Response from TTS inference.
#[derive(Debug, Clone)]
pub struct TTSResponse {
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub generation_time_ms: u64,
    pub model: String,
    pub seed_used: u64,
}

/// Trait for TTS inference backends.
pub trait InferenceEngine: Send + Sync {
    fn synthesize(&mut self, req: &TTSRequest) -> Result<TTSResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<()>;
    fn unload(&mut self) {}
    fn set_on_progress(&mut self, _callback: ProgressCallback) {}
}
