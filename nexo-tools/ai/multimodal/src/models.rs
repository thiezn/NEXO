use serde::Serialize;

#[derive(Debug, Clone)]
pub struct DescriptionConfig {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DescriptionResult {
    pub text: String,
    pub model: String,
    pub prompt_used: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}
