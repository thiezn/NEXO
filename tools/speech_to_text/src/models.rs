use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
    pub model: String,
    pub duration_secs: f64,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub model: String,
    pub language: String,
    pub translate: bool,
    pub timestamps: bool,
}
