use super::WhisperModel;
use crate::config::WhisperModelPaths;
use candle_transformers::models::whisper::{self, audio as whisper_audio, Config};
use local_inference_helpers::candle_core::{Device, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use tokenizers::Tokenizer;

/// Load a standard (FP32) Whisper model from safetensors files.
pub fn load_model(paths: &WhisperModelPaths, device: &Device) -> anyhow::Result<WhisperModel> {
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&paths.config_json)?)?;

    tracing::info!(
        d_model = config.d_model,
        encoder_layers = config.encoder_layers,
        decoder_layers = config.decoder_layers,
        "loading whisper model"
    );

    let mut model_files = vec![paths.model_file.clone()];
    model_files.extend(paths.model_shards.iter().cloned());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&model_files, whisper::DTYPE, device)?
    };

    let model = whisper::model::Whisper::load(&vb, config)?;
    Ok(WhisperModel::Standard(model))
}

/// Load model config from config.json.
pub fn load_config(paths: &WhisperModelPaths) -> anyhow::Result<Config> {
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&paths.config_json)?)?;
    Ok(config)
}

/// Load tokenizer from tokenizer.json.
pub fn load_tokenizer(paths: &WhisperModelPaths) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file(&paths.tokenizer)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

/// Compute mel spectrogram from PCM samples using pre-computed filters.
pub fn pcm_to_mel_with_filters(
    config: &Config,
    samples: &[f32],
    mel_filters: &[f32],
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mel = whisper_audio::pcm_to_mel(config, samples, mel_filters);
    let mel_len = mel.len() / config.num_mel_bins;
    let mel = Tensor::from_vec(mel, (1, config.num_mel_bins, mel_len), device)?;
    Ok(mel)
}

/// Generate mel filter bank weights for the given number of bins.
///
/// Call once at startup and reuse for all chunks. Uses standard mel-scale
/// with Whisper parameters: 16kHz sample rate, N_FFT=400.
pub fn mel_filters_for(num_mel_bins: usize) -> Vec<f32> {
    let n_freqs = whisper::N_FFT / 2 + 1;
    let sample_rate = whisper::SAMPLE_RATE as f32;

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| (n_freqs as f32 * hz / (sample_rate / 2.0)).floor() as usize)
        .collect();

    let mut filters = vec![0.0f32; num_mel_bins * n_freqs];
    for i in 0..num_mel_bins {
        let start = bin_points[i];
        let center = bin_points[i + 1];
        let end = bin_points[i + 2];

        for j in start..center {
            if center > start && j < n_freqs {
                filters[i * n_freqs + j] = (j - start) as f32 / (center - start) as f32;
            }
        }
        for j in center..end {
            if end > center && j < n_freqs {
                filters[i * n_freqs + j] = (end - j) as f32 / (end - center) as f32;
            }
        }
    }

    // Normalize each filter
    for i in 0..num_mel_bins {
        let sum: f32 = (0..n_freqs).map(|j| filters[i * n_freqs + j]).sum();
        if sum > 0.0 {
            for j in 0..n_freqs {
                filters[i * n_freqs + j] /= sum;
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
