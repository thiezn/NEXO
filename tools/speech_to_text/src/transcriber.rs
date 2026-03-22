use crate::audio;
use crate::config::{AppConfig, WhisperModelPaths};
use crate::inference::{decoder, engine, language};
use crate::models::{Segment, TranscriptionConfig, TranscriptionResult};
use local_inference_helpers::device::create_device;
use std::path::Path;
use std::time::Instant;

/// Transcribe an audio file to text using a Whisper model.
pub fn transcribe(
    config: &TranscriptionConfig,
    audio_path: &Path,
    app_config: &AppConfig,
) -> anyhow::Result<TranscriptionResult> {
    let start = Instant::now();
    let model_name = &config.model;

    let paths = WhisperModelPaths::resolve(model_name, app_config).ok_or_else(|| {
        anyhow::anyhow!(
            "model '{model_name}' not found in config. Run: speech_to_text pull {model_name}"
        )
    })?;

    crate::config::validate_paths(&paths)?;

    // Load and decode audio
    tracing::info!(path = %audio_path.display(), "loading audio");
    let audio_data = audio::decode::load_audio(audio_path)?;

    // Resample to 16kHz
    let samples = audio::resample::resample_to_16khz(audio_data)?;
    let total_duration = samples.len() as f64 / 16_000.0;
    tracing::info!(duration_secs = total_duration, "audio ready");

    // Set up device
    let device = create_device(|info| tracing::info!("{info}"))?;

    // Load model
    tracing::info!(model = model_name, "loading model");
    let whisper_config = engine::load_config(&paths)?;
    let mel_filters = engine::mel_filters_for(whisper_config.num_mel_bins);
    let tokenizer = engine::load_tokenizer(&paths)?;
    let mut model = engine::load_model(&paths, &device)?;
    tracing::info!("model loaded");

    // Determine language
    let language = if config.language == "auto" {
        let mel = engine::pcm_to_mel_with_filters(
            &whisper_config,
            &samples[..samples.len().min(candle_transformers::models::whisper::N_SAMPLES)],
            &mel_filters,
            &device,
        )?;
        let audio_features = model.encoder_forward(&mel, true)?;
        let (detected, prob) =
            language::detect_language(&mut model, &audio_features, &tokenizer, &device)?;
        tracing::info!(language = %detected, probability = prob, "auto-detected language");
        detected
    } else {
        config.language.clone()
    };

    // Transcribe chunks
    let mut chunker = audio::chunk::AudioChunker::new(samples);
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut all_text = String::new();
    let mut chunk_idx = 0;

    while let Some((chunk, chunk_offset)) = chunker.next_chunk() {
        tracing::info!(
            chunk = chunk_idx,
            offset_secs = chunk_offset,
            "processing chunk"
        );

        let mel =
            engine::pcm_to_mel_with_filters(&whisper_config, chunk, &mel_filters, &device)?;
        let audio_features = model.encoder_forward(&mel, true)?;

        let output = decoder::decode_chunk(
            &mut model,
            &audio_features,
            &tokenizer,
            &language,
            config.translate,
            config.timestamps,
            &device,
            chunk_offset,
        )?;

        if output.no_speech_prob > candle_transformers::models::whisper::NO_SPEECH_THRESHOLD {
            tracing::debug!(chunk = chunk_idx, "no speech detected, skipping");
        }

        for seg in &output.segments {
            if !seg.text.is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(&seg.text);
            }
        }
        all_segments.extend(output.segments);

        // Seek based on last timestamp from decoder
        if let Some(last_ts) = decoder::last_timestamp(&output.tokens) {
            chunker.seek_to_time(chunk_offset + last_ts);
        }

        chunk_idx += 1;
    }

    let inference_time = start.elapsed().as_millis() as u64;
    tracing::info!(
        inference_time_ms = inference_time,
        segments = all_segments.len(),
        "transcription complete"
    );

    Ok(TranscriptionResult {
        text: all_text,
        segments: all_segments,
        language,
        model: model_name.clone(),
        duration_secs: total_duration,
        inference_time_ms: inference_time,
    })
}
