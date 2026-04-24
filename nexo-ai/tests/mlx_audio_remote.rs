//! End-to-end integration tests for the MLX Audio remote provider.
//!
//! These tests exercise the manifest -> coordinator -> factory -> managed
//! provider server -> OpenAI speech adapter path for remote STT and TTS models.
//! They are `#[ignore]` by default since they require:
//! - The `mlx` feature
//! - A Python venv with `mlx-audio` and its speech dependencies installed
//! - macOS with Apple Silicon
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "mlx")]

mod common;

use std::net::TcpListener;
use std::path::Path;

use ntest::timeout;
use serial_test::serial;

use common::init_tracing;
use nexo_ai::api::types::{ListenRequest, TalkRequest};

fn load_test_audio() -> (Vec<f32>, u32) {
    let wav_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("datasets/audio/monkeyinmypocket.wav");
    assert!(
        wav_path.exists(),
        "test audio not found at {}",
        wav_path.display()
    );
    let buf = nexo_ai::audio::load_file(&wav_path).expect("failed to decode test audio");
    let mono = buf.to_mono();
    (mono.samples, mono.sample_rate)
}

fn reserve_local_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to reserve an ephemeral port")
        .local_addr()
        .expect("failed to inspect reserved port")
        .port()
}

fn with_tokio_runtime(test: impl FnOnce()) {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime for remote model test")
        .block_on(async move { test() });
}

fn make_mlx_audio_coordinator() -> nexo_ai::coordinator::Coordinator {
    let mut config = nexo_ai::config::CoordinatorConfig::default();
    config.startup_categories.clear();
    config.mlx_audio_port = Some(reserve_local_port());
    config.mlx_audio_hf_endpoint = Some("https://hf-mirror.com".to_string());
    if let Ok(venv_path) = std::env::var("VIRTUAL_ENV") {
        config.mlx_audio_venv_path = Some(venv_path);
    }
    nexo_ai::coordinator::Coordinator::new(config)
}

#[test]
#[ignore]
#[serial]
#[timeout(1_800_000)]
fn test_mlx_whisper_large_v3_turbo_asr_fp16_remote() {
    init_tracing();
    with_tokio_runtime(|| {
        let model_name = "mlx-whisper-large-v3-turbo-asr-fp16";
        let mut coordinator = make_mlx_audio_coordinator();

        coordinator
            .load_model(model_name)
            .expect("failed to load remote whisper model through coordinator");
        assert!(
            coordinator.is_model_loaded(model_name),
            "coordinator should report the remote whisper model as loaded"
        );

        let (pcm_samples, sample_rate) = load_test_audio();
        let request = ListenRequest {
            pcm_samples,
            sample_rate,
            language: None,
        };

        let response = {
            let model = coordinator
                .model_mut(model_name)
                .expect("remote whisper model slot should exist");
            let listen = model
                .as_listen()
                .expect("remote whisper model should support listen");
            listen
                .transcribe(&request)
                .expect("remote whisper transcription failed")
        };

        eprintln!("remote whisper transcription: {:?}", response.text);
        assert!(
            !response.text.trim().is_empty(),
            "remote whisper transcription returned empty text"
        );
        assert!(
            response.text.chars().any(|ch| ch.is_alphabetic()),
            "remote whisper transcription should contain spoken words"
        );
        assert!(
            response.inference_time_ms > 0,
            "remote whisper transcription should report latency"
        );

        coordinator.unload_all();
        assert_eq!(coordinator.loaded_model_count(), 0);
    });
}

#[test]
#[ignore]
#[serial]
#[timeout(1_800_000)]
fn test_mlx_voxtral_4b_tts_2603_bf16_remote() {
    init_tracing();
    with_tokio_runtime(|| {
        let model_name = "mlx-voxtral-4b-tts-2603-bf16";
        let mut coordinator = make_mlx_audio_coordinator();

        coordinator
            .load_model(model_name)
            .expect("failed to load remote voxtral model through coordinator");
        assert!(
            coordinator.is_model_loaded(model_name),
            "coordinator should report the remote voxtral model as loaded"
        );

        let request = TalkRequest {
            text: "This is an end-to-end Voxtral speech synthesis test from nexo-ai.".into(),
            voice_description: "A clear, natural speaker with a steady pace.".into(),
            max_tokens: 512,
            temperature: 0.7,
            seed: 7,
        };

        let response = {
            let model = coordinator
                .model_mut(model_name)
                .expect("remote voxtral model slot should exist");
            let talk = model
                .as_talk()
                .expect("remote voxtral model should support talk");
            talk.synthesize(&request)
                .expect("remote voxtral synthesis failed")
        };

        eprintln!(
            "remote voxtral audio: {} samples @ {} Hz in {} ms",
            response.pcm_samples.len(),
            response.sample_rate,
            response.inference_time_ms,
        );
        assert!(
            response.sample_rate > 0,
            "remote voxtral sample rate must be positive"
        );
        assert!(
            !response.pcm_samples.is_empty(),
            "remote voxtral synthesis returned no PCM samples"
        );
        assert!(
            response
                .pcm_samples
                .iter()
                .any(|sample| sample.abs() > f32::EPSILON),
            "remote voxtral synthesis returned only silence"
        );
        assert!(
            response.inference_time_ms > 0,
            "remote voxtral synthesis should report latency"
        );

        coordinator.unload_all();
        assert_eq!(coordinator.loaded_model_count(), 0);
    });
}
