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
    // candle-transformers pads mel beyond N_FRAMES; truncate to fit encoder positional embeddings
    let mel = mel.narrow(2, 0, mel_len.min(whisper::N_FRAMES))?;
    Ok(mel)
}

/// Generate mel filter bank weights matching librosa's `mel(sr=16000, n_fft=400, n_mels, norm="slaney")`.
///
/// Call once at startup and reuse for all chunks. Uses Slaney normalization
/// to produce filters compatible with OpenAI Whisper's expected mel spectrogram.
pub fn mel_filters_for(num_mel_bins: usize) -> Vec<f32> {
    let n_freqs = whisper::N_FFT / 2 + 1; // 201
    let sample_rate = whisper::SAMPLE_RATE as f32;
    let fmax = sample_rate / 2.0;

    // FFT bin center frequencies: [0, sr/n_fft, 2*sr/n_fft, ..., sr/2]
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / whisper::N_FFT as f32)
        .collect();

    // Mel-spaced center frequencies (num_mel_bins + 2 points including edges)
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(fmax);
    let mel_f: Vec<f32> = (0..num_mel_bins + 2)
        .map(|i| mel_to_hz(mel_low + (mel_high - mel_low) * i as f32 / (num_mel_bins + 1) as f32))
        .collect();

    let mut filters = vec![0.0f32; num_mel_bins * n_freqs];
    for i in 0..num_mel_bins {
        let f_low = mel_f[i];
        let f_center = mel_f[i + 1];
        let f_high = mel_f[i + 2];
        let fdiff_low = f_center - f_low;
        let fdiff_high = f_high - f_center;

        for (j, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - f_low) / fdiff_low;
            let upper = (f_high - freq) / fdiff_high;
            let val = lower.min(upper).max(0.0);
            filters[i * n_freqs + j] = val;
        }

        // Slaney normalization: 2 / (high_hz - low_hz)
        let enorm = 2.0 / (f_high - f_low);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm;
        }
    }

    filters
}

/// Slaney mel scale (matching librosa default htk=False).
/// Linear below 1000 Hz, logarithmic above.
fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
    const LOGSTEP: f32 = 0.06875177; // ln(6.4) / 27

    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Inverse Slaney mel scale.
fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.06875177;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::whisper::{N_FRAMES, N_SAMPLES};
    use local_inference_helpers::candle_core::Device;

    fn test_config(num_mel_bins: usize) -> Config {
        Config {
            num_mel_bins,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            vocab_size: 51866,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 4,
            suppress_tokens: vec![],
        }
    }

    #[test]
    fn mel_shape_full_chunk() {
        let config = test_config(128);
        let samples = vec![0.0f32; N_SAMPLES];
        let filters = mel_filters_for(config.num_mel_bins);
        let mel = pcm_to_mel_with_filters(&config, &samples, &filters, &Device::Cpu).unwrap();
        let dims = mel.dims3().unwrap();
        assert_eq!(dims, (1, 128, N_FRAMES));
    }

    #[test]
    fn mel_shape_short_audio() {
        let config = test_config(128);
        let samples = vec![0.0f32; 16_000]; // 1 second
        let filters = mel_filters_for(config.num_mel_bins);
        let mel = pcm_to_mel_with_filters(&config, &samples, &filters, &Device::Cpu).unwrap();
        let (_, _, mel_len) = mel.dims3().unwrap();
        assert!(mel_len <= N_FRAMES, "mel_len {mel_len} exceeds N_FRAMES {N_FRAMES}");
    }
}
