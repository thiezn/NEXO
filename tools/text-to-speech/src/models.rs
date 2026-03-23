use serde::Serialize;

/// Audio output format.
#[derive(Debug, Clone, Copy, Default)]
pub enum AudioFormat {
    #[default]
    Wav,
}

impl AudioFormat {
    pub fn extension(&self) -> &str {
        match self {
            Self::Wav => "wav",
        }
    }
}

/// Configuration for the TTS synthesis pipeline.
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    pub model: String,
    pub text: String,
    pub description: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
    pub output_format: AudioFormat,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            model: "parler-mini".to_string(),
            text: String::new(),
            description: "A clear, natural speaking voice at a moderate pace.".to_string(),
            max_tokens: None,
            temperature: None,
            seed: None,
            output_format: AudioFormat::default(),
        }
    }
}

/// Result of the TTS synthesis pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisResult {
    pub text_used: String,
    pub description_used: String,
    pub audio: GeneratedAudio,
}

/// A single generated audio clip.
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedAudio {
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub seed: u64,
    pub generation_time_ms: u64,
    #[serde(skip)]
    pub pcm_data: Vec<f32>,
}
