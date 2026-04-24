use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper;
use candle_transformers::models::whisper::model::Whisper;

use crate::audio::AudioBuffer;
use crate::api::types::{ListenRequest, ListenResponse};
use crate::models::support::weights::find_safetensor_files;

use super::decode::{self, WhisperTokens};
use super::mel;

pub struct LoadedState {
    pub model: Whisper,
    pub tokenizer: tokenizers::Tokenizer,
    pub whisper_tokens: WhisperTokens,
    pub mel_filters: Vec<f32>,
    pub device: Device,
    pub suppress_mask: Vec<bool>,
}

/// Load a Whisper model from the given directory.
///
/// Expects `config.json`, `tokenizer.json`, and one or more `.safetensors` files.
pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let start = Instant::now();

    let device = crate::device::create_device()?;
    let dtype = crate::device::gpu_dtype(&device);
    tracing::info!("device ready in {:.1}s", start.elapsed().as_secs_f64());

    // Config
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", config_path.display()))?;
    let config: whisper::Config = serde_json::from_str(&config_str)?;
    tracing::info!(
        "config: d_model={}, encoder_layers={}, decoder_layers={}, mel_bins={}",
        config.d_model,
        config.encoder_layers,
        config.decoder_layers,
        config.num_mel_bins
    );

    // Weights
    let safetensor_files = find_safetensor_files(model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)? };
    let model = Whisper::load(&vb, config.clone())?;
    tracing::info!(
        "model loaded in {:.1}s ({} safetensor file(s))",
        start.elapsed().as_secs_f64(),
        safetensor_files.len()
    );

    // Tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let whisper_tokens = WhisperTokens::new(&tokenizer)?;
    tracing::info!(
        "tokenizer loaded (sot={}, eot={}, timestamp_base={})",
        whisper_tokens.sot,
        whisper_tokens.eot,
        whisper_tokens.timestamp_base
    );

    // Mel filterbank
    let mel_filters = mel::compute_mel_filters(config.num_mel_bins);

    // Suppress mask
    let suppress_mask = decode::build_suppress_mask(&config.suppress_tokens, config.vocab_size);

    tracing::info!(
        "Whisper fully loaded in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    Ok(LoadedState {
        model,
        tokenizer,
        whisper_tokens,
        mel_filters,
        device,
        suppress_mask,
    })
}

/// Run speech-to-text transcription on an already-loaded model.
pub fn transcribe(state: &mut LoadedState, request: &ListenRequest) -> Result<ListenResponse> {
    let start = Instant::now();

    // Resample to 16 kHz mono
    let samples = prepare_audio(&request.pcm_samples, request.sample_rate)?;

    // Chunk into 30-second segments
    let chunks = chunk_audio(&samples);
    tracing::info!(
        "audio: {:.1}s, {} chunk(s)",
        samples.len() as f64 / whisper::SAMPLE_RATE as f64,
        chunks.len()
    );

    let mut all_text = String::new();
    let mut all_segments = Vec::new();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let chunk_offset_ms = chunk_idx as u64 * whisper::CHUNK_LENGTH as u64 * 1000;

        let (text, segments) =
            transcribe_chunk(state, chunk, request.language.as_deref(), chunk_offset_ms)?;

        if !text.is_empty() {
            if !all_text.is_empty() {
                all_text.push(' ');
            }
            all_text.push_str(&text);
        }
        all_segments.extend(segments);
    }

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        "transcribed {:.1}s of audio in {:.1}s",
        samples.len() as f64 / whisper::SAMPLE_RATE as f64,
        inference_time_ms as f64 / 1000.0
    );

    Ok(ListenResponse {
        text: all_text,
        segments: all_segments,
        language: request.language.clone(),
        inference_time_ms,
    })
}

/// Resample input audio to 16 kHz mono.
fn prepare_audio(pcm_samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    let buf = AudioBuffer::new(pcm_samples.to_vec(), sample_rate, 1);
    let mono = buf.to_mono();

    if mono.sample_rate == whisper::SAMPLE_RATE as u32 {
        return Ok(mono.samples);
    }

    let resampled = crate::audio::resample(mono, whisper::SAMPLE_RATE as u32)?;
    Ok(resampled.samples)
}

/// Split audio into 30-second chunks, zero-padding the last chunk.
fn chunk_audio(samples: &[f32]) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return vec![vec![0.0; whisper::N_SAMPLES]];
    }

    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < samples.len() {
        let end = (offset + whisper::N_SAMPLES).min(samples.len());
        let mut chunk = vec![0.0; whisper::N_SAMPLES];
        chunk[..end - offset].copy_from_slice(&samples[offset..end]);
        chunks.push(chunk);
        offset += whisper::N_SAMPLES;
    }

    chunks
}

/// Transcribe a single 30-second chunk.
fn transcribe_chunk(
    state: &mut LoadedState,
    chunk: &[f32],
    language: Option<&str>,
    chunk_offset_ms: u64,
) -> Result<(String, Vec<crate::api::types::TranscriptionSegment>)> {
    let config = &state.model.config;

    // Compute mel spectrogram and truncate to N_FRAMES (3000).
    // Candle's pcm_to_mel pads beyond 3000 frames; after the encoder's conv2
    // (stride=2) the extra frames would exceed max_source_positions (1500).
    let mel = whisper::audio::pcm_to_mel(config, chunk, &state.mel_filters);
    let full_n_frames = mel.len() / config.num_mel_bins;
    let n_frames = full_n_frames.min(whisper::N_FRAMES);

    let mel = if full_n_frames > n_frames {
        let mut truncated = Vec::with_capacity(config.num_mel_bins * n_frames);
        for m in 0..config.num_mel_bins {
            let start = m * full_n_frames;
            truncated.extend_from_slice(&mel[start..start + n_frames]);
        }
        truncated
    } else {
        mel
    };

    let mel_tensor = Tensor::from_vec(mel, (1, config.num_mel_bins, n_frames), &state.device)?;

    // Encode audio
    let encoder_output = state.model.encoder.forward(&mel_tensor, true)?;

    // Decode tokens autoregressively
    let initial_tokens = state
        .whisper_tokens
        .initial_tokens(&state.tokenizer, language);
    let max_decode_len = config.max_target_positions / 2;
    let mut token_ids: Vec<u32> = initial_tokens;
    let mut decoded_tokens: Vec<u32> = Vec::new();

    for step in 0..max_decode_len {
        let token_tensor =
            Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &state.device)?;

        let flush = step == 0;
        let decoder_output = state
            .model
            .decoder
            .forward(&token_tensor, &encoder_output, flush)?;

        // Get logits for the last token position
        let seq_len = decoder_output.dim(1)?;
        let last_hidden = decoder_output.i((.., seq_len - 1..))?;
        let logits = state.model.decoder.final_linear(&last_hidden)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply suppress mask and argmax
        let next_token = argmax_with_suppress(&logits_vec, &state.suppress_mask);

        if next_token == state.whisper_tokens.eot {
            break;
        }

        decoded_tokens.push(next_token);
        // Accumulate full token history — the decoder needs the complete sequence
        token_ids.push(next_token);
    }

    let n_timestamps = decoded_tokens
        .iter()
        .filter(|&&t| state.whisper_tokens.is_timestamp(t))
        .count();
    tracing::info!(
        "decoded {} tokens ({} text, {} timestamps)",
        decoded_tokens.len(),
        decoded_tokens.len() - n_timestamps,
        n_timestamps,
    );

    // Reset KV cache for the next chunk
    state.model.reset_kv_cache();

    // Decode tokens to text with timestamps
    let (text, segments) = decode::decode_with_timestamps(
        &state.whisper_tokens,
        &state.tokenizer,
        &decoded_tokens,
        chunk_offset_ms,
    );

    Ok((text, segments))
}

/// Greedy argmax with token suppression.
fn argmax_with_suppress(logits: &[f32], suppress_mask: &[bool]) -> u32 {
    let mut best_token = 0u32;
    let mut best_score = f32::NEG_INFINITY;

    for (i, &logit) in logits.iter().enumerate() {
        if i < suppress_mask.len() && suppress_mask[i] {
            continue;
        }
        if logit > best_score {
            best_score = logit;
            best_token = i as u32;
        }
    }

    best_token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_audio_short() {
        let samples = vec![1.0; 1000];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), whisper::N_SAMPLES);
        assert_eq!(chunks[0][0], 1.0);
        assert_eq!(chunks[0][999], 1.0);
        assert_eq!(chunks[0][1000], 0.0); // zero-padded
    }

    #[test]
    fn chunk_audio_exact_30s() {
        let samples = vec![0.5; whisper::N_SAMPLES];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), whisper::N_SAMPLES);
    }

    #[test]
    fn chunk_audio_over_30s() {
        let samples = vec![0.5; whisper::N_SAMPLES + 1000];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1][999], 0.5);
        assert_eq!(chunks[1][1000], 0.0);
    }

    #[test]
    fn chunk_audio_empty() {
        let chunks = chunk_audio(&[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), whisper::N_SAMPLES);
        assert!(chunks[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn argmax_basic() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mask = vec![false; 5];
        assert_eq!(argmax_with_suppress(&logits, &mask), 3);
    }

    #[test]
    fn argmax_with_suppression() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mask = vec![false, false, false, true, false]; // suppress token 3
        assert_eq!(argmax_with_suppress(&logits, &mask), 1);
    }
}
