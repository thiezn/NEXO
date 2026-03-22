pub(crate) mod encoders;
pub mod factory;
pub mod image;
pub mod img_utils;
pub(crate) mod pipelines;

use anyhow::Result;
use local_inference_helpers::progress::ProgressCallback;

use crate::models::OutputFormat;

/// Controls how model components are loaded during inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Load all components at once, keep hot (server mode).
    #[default]
    Eager,
    /// Load-use-drop per component, minimizing peak memory (CLI one-shot mode).
    Sequential,
}

/// Request for the inference engine.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: u64,
    pub batch_size: u32,
    pub output_format: OutputFormat,
}

/// Single generated image data.
#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub format: OutputFormat,
    pub width: u32,
    pub height: u32,
    pub index: u32,
}

/// Response from the inference engine.
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub images: Vec<ImageData>,
    pub generation_time_ms: u64,
    pub model: String,
    pub seed_used: u64,
}

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<()>;
    fn unload(&mut self) {}
    fn set_on_progress(&mut self, _callback: ProgressCallback) {}
}
