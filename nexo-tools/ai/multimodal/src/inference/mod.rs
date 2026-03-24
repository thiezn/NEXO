pub mod engine;
pub mod linear_attn;
pub mod text;

pub struct TextRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

pub struct TextResponse {
    pub text: String,
    pub tokens_generated: usize,
}
