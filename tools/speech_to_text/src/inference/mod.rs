pub mod decoder;
pub mod engine;
pub mod language;

use candle_transformers::models::whisper::{model, quantized_model, Config};
use local_inference_helpers::candle_core::{Result, Tensor};

/// Unified wrapper around standard and quantized Whisper models.
pub enum WhisperModel {
    Standard(model::Whisper),
    Quantized(quantized_model::Whisper),
}

impl WhisperModel {
    pub fn config(&self) -> &Config {
        match self {
            Self::Standard(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, mel: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.encoder.forward(mel, flush_kv_cache),
            Self::Quantized(m) => m.encoder.forward(mel, flush_kv_cache),
        }
    }

    pub fn decoder_forward(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Standard(m) => {
                let x = m.decoder.forward(tokens, audio_features, flush_kv_cache)?;
                m.decoder.final_linear(&x)
            }
            Self::Quantized(m) => {
                let x = m.decoder.forward(tokens, audio_features, flush_kv_cache)?;
                m.decoder.final_linear(&x)
            }
        }
    }

    pub fn reset_kv_cache(&mut self) {
        match self {
            Self::Standard(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}
