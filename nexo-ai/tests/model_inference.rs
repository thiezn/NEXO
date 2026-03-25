//! Integration tests for nexo-ai model inference.
//!
//! These tests load real models, run inference, and validate output.
//! They are `#[ignore]` by default since they require downloaded models and hardware.
//!
//! Run all:  `cargo test -p nexo-ai --test model_inference -- --ignored`
//! Run one:  `cargo test -p nexo-ai --test model_inference -- --ignored test_whisper_large_v3_turbo`

use std::path::Path;
use std::sync::Once;

use ntest::timeout;
use serial_test::serial;

use nexo_ai::download::manifest::storage_path;
use nexo_ai::download::paths::{default_models_dir, model_storage_dir};
use nexo_ai::registry::manifest::find_manifest;
use nexo_ai::shared::model_traits::ModelInfo;
use nexo_ai::shared::types::*;

// ── Tracing ─────────────────────────────────────────────────────────────────

static INIT_TRACING: Once = Once::new();

/// Initialize tracing for integration tests. Debug level for nexo_ai, info for everything else.
/// Call at the start of each test; `Once` ensures it only runs once per test binary.
fn init_tracing() {
    INIT_TRACING.call_once(|| {
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,nexo_ai=debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Resolve model directory and memory estimate from the manifest registry.
/// Panics with a clear download instruction if the model is not downloaded.
fn resolve_model(model_name: &str) -> (std::path::PathBuf, u64) {
    init_tracing();

    let manifest = find_manifest(model_name)
        .unwrap_or_else(|| panic!("unknown model '{model_name}' in manifest registry"));

    let dir = model_storage_dir(model_name);
    if !dir.exists() {
        panic!(
            "\n\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  MODEL NOT DOWNLOADED: {:<37} ║\n\
             ╠══════════════════════════════════════════════════════════════╣\n\
             ║  Expected directory:                                        ║\n\
             ║    {}\n\
             ║                                                              ║\n\
             ║  Download with:                                              ║\n\
             ║    cargo run -p nexo-ai --features cli -- pull {}\n\
             ╚══════════════════════════════════════════════════════════════╝\n",
            model_name,
            dir.display(),
            model_name,
        );
    }

    // Verify all expected files from the manifest are present
    let models_dir = default_models_dir();
    let missing: Vec<_> = manifest
        .manifest
        .files
        .iter()
        .filter(|f| {
            let path = models_dir.join(storage_path(&manifest.manifest, f));
            !path.exists()
        })
        .map(|f| f.hf_filename.as_str())
        .collect();

    if !missing.is_empty() {
        panic!(
            "\n\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  MODEL INCOMPLETE: {:<40} ║\n\
             ╠══════════════════════════════════════════════════════════════╣\n\
             ║  Missing files:                                              ║\n\
             {}\
             ║                                                              ║\n\
             ║  Re-download with:                                           ║\n\
             ║    cargo run -p nexo-ai --features cli -- pull {} --force\n\
             ╚══════════════════════════════════════════════════════════════╝\n",
            model_name,
            missing
                .iter()
                .map(|f| format!("║    - {f}\n"))
                .collect::<String>(),
            model_name,
        );
    }

    let memory_bytes = (manifest.manifest.size_gb * 1_000_000_000.0) as u64;
    (dir, memory_bytes)
}

/// Load the test speech WAV file and return PCM samples + sample rate.
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

// ── Whisper (Listen) ─────────────────────────────────────────────────────────

macro_rules! listen_test {
    ($name:ident, $model_name:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::listen::whisper::WhisperModel::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let (samples, sample_rate) = load_test_audio();
            let request = ListenRequest {
                pcm_samples: samples,
                sample_rate,
                language: None,
            };

            let listen = model.as_listen().expect("should be a listen model");
            let response = listen.transcribe(&request).expect("transcription failed");

            eprintln!("transcription: {:?}", response.text);
            assert!(
                !response.text.is_empty(),
                "transcription returned empty text"
            );

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

listen_test!(test_whisper_large_v3_turbo, "whisper-large-v3-turbo");
listen_test!(test_whisper_large_v3, "whisper-large-v3");
listen_test!(test_distil_large_v3, "distil-large-v3");

// ── Parler (Talk) ────────────────────────────────────────────────────────────

fn parler_request() -> TalkRequest {
    TalkRequest {
        text: "Hello world.".into(),
        voice_description: "A warm female voice.".into(),
        max_tokens: 50,
        temperature: 1.0,
        seed: 42,
    }
}

macro_rules! talk_test {
    ($name:ident, $model_name:expr, $timeout:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout($timeout)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::talk::parler::ParlerTtsModel::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let talk = model.as_talk().expect("should be a talk model");
            let response = talk
                .synthesize(&parler_request())
                .expect("synthesis failed");

            eprintln!(
                "generated {} samples at {} Hz ({:.1}s)",
                response.pcm_samples.len(),
                response.sample_rate,
                response.pcm_samples.len() as f64 / response.sample_rate as f64
            );
            assert!(
                !response.pcm_samples.is_empty(),
                "synthesis returned empty PCM"
            );
            assert!(response.sample_rate > 0, "invalid sample rate");

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

talk_test!(test_parler_mini, "parler-mini", 600_000);
talk_test!(test_parler_large, "parler-large", 900_000);

// ── Parler (Performance) ────────────────────────────────────────────────────

macro_rules! talk_perf_test {
    ($name:ident, $model_name:expr, $max_seconds:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::talk::parler::ParlerTtsModel::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let talk = model.as_talk().expect("should be a talk model");
            let request = TalkRequest {
                text: "Hello world.".into(),
                voice_description: "A warm female voice.".into(),
                max_tokens: 50,
                temperature: 1.0,
                seed: 42,
            };
            let response = talk.synthesize(&request).expect("synthesis failed");

            let audio_duration = response.pcm_samples.len() as f64 / response.sample_rate as f64;
            let inference_secs = response.inference_time_ms as f64 / 1000.0;
            let rtf = inference_secs / audio_duration;

            eprintln!(
                "PERF: {} — {:.1}s audio in {:.1}s = {:.1}x realtime (max: {:.0}s)",
                $model_name, audio_duration, inference_secs, rtf, $max_seconds as f64,
            );

            assert!(
                inference_secs <= $max_seconds as f64,
                "Performance regression: {:.1}s > {:.0}s maximum for {}",
                inference_secs, $max_seconds as f64, $model_name,
            );

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

talk_perf_test!(test_parler_mini_perf, "parler-mini", 60);
talk_perf_test!(test_parler_large_perf, "parler-large", 120);

// ── Flux (Imagine) ───────────────────────────────────────────────────────────

fn flux_request() -> ImagineRequest {
    ImagineRequest {
        prompt: "a red circle on white background".into(),
        width: 256,
        height: 256,
        steps: 1,
        guidance: 3.5,
        seed: 42,
        batch_size: 1,
    }
}

macro_rules! imagine_test {
    ($name:ident, $model_name:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::imagine::flux::FluxModel::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let imagine = model.as_imagine().expect("should be an imagine model");
            let response = imagine
                .imagine(&flux_request())
                .expect("image generation failed");

            eprintln!("generated {} image(s)", response.images.len());
            assert!(
                !response.images.is_empty(),
                "image generation returned no images"
            );

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

imagine_test!(test_flux_2_klein_4b, "flux-2-klein-4b");
imagine_test!(test_flux_2_klein_9b, "flux-2-klein-9b");
imagine_test!(test_flux_2_dev, "flux-2-dev");

// ── Gemma 3 (Chat) ─────────────────────────────────────────────────────────

macro_rules! chat_test {
    ($name:ident, $model_name:expr, $model_type:path) => {
        chat_test!($name, $model_name, $model_type, 32);
    };
    ($name:ident, $model_name:expr, $model_type:path, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");
            let request = ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "What is 2+2? Answer with just the number.".into(),
                }],
                max_tokens: $max_tokens,
                temperature: 0.1,
                top_p: 0.9,
            };
            let response = chat.chat(&request).expect("chat failed");

            eprintln!("response: {:?}", response.text);
            assert!(!response.text.is_empty(), "chat returned empty text");
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

chat_test!(test_gemma_3_4b_it_chat, "gemma-3-4b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
chat_test!(test_gemma_3_12b_it_chat, "gemma-3-12b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
chat_test!(test_gemma_3_27b_it_chat, "gemma-3-27b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);

// ── Gemma 3 (Performance) ──────────────────────────────────────────────────

macro_rules! perf_test {
    ($name:ident, $model_name:expr, $min_tok_per_sec:expr, $model_type:path) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");

            // Warmup: prime Metal shader compilation
            let warmup = ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Hi".into(),
                }],
                max_tokens: 2,
                temperature: 0.1,
                top_p: 0.9,
            };
            let _ = chat.chat(&warmup);

            // Benchmark
            let request = ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Write a short paragraph about the history of computing.".into(),
                }],
                max_tokens: 128,
                temperature: 0.1,
                top_p: 0.9,
            };
            let response = chat.chat(&request).expect("chat failed");

            let tok_per_sec = if response.inference_time_ms > 0 {
                response.tokens_generated as f64 / (response.inference_time_ms as f64 / 1000.0)
            } else {
                0.0
            };

            eprintln!(
                "PERF: {} — {} tokens in {}ms = {:.1} tok/s (min: {:.1})",
                $model_name,
                response.tokens_generated,
                response.inference_time_ms,
                tok_per_sec,
                $min_tok_per_sec as f64,
            );

            assert!(
                tok_per_sec >= $min_tok_per_sec as f64,
                "Performance regression: {:.1} tok/s < {:.1} tok/s minimum for {}",
                tok_per_sec,
                $min_tok_per_sec as f64,
                $model_name,
            );

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

perf_test!(test_gemma_3_4b_it_perf, "gemma-3-4b-it", 10.0, nexo_ai::models::multipurpose::gemma3::Gemma3Model);
perf_test!(test_gemma_3_12b_it_perf, "gemma-3-12b-it", 5.0, nexo_ai::models::multipurpose::gemma3::Gemma3Model);
perf_test!(test_gemma_3_27b_it_perf, "gemma-3-27b-it", 2.0, nexo_ai::models::multipurpose::gemma3::Gemma3Model);

// ── Gemma 3 (Tool) ─────────────────────────────────────────────────────────

macro_rules! tool_test {
    ($name:ident, $model_name:expr, $model_type:path) => {
        tool_test!($name, $model_name, $model_type, 128);
    };
    ($name:ident, $model_name:expr, $model_type:path, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let tool = model.as_tool().expect("should be a tool model");
            let request = ToolCallRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "What is the weather in Amsterdam?".into(),
                }],
                tools: vec![nexo_tool_spec::tool::ToolSpec {
                    name: "get_weather".into(),
                    description: "Get the current weather for a city".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    }),
                }],
                max_tokens: $max_tokens,
                temperature: 0.1,
            };
            let response = tool.call_tools(&request).expect("tool call failed");

            eprintln!("tool calls: {:?}", response.tool_calls);
            eprintln!("reasoning: {:?}", response.reasoning);
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

tool_test!(test_gemma_3_4b_it_tool, "gemma-3-4b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
tool_test!(test_gemma_3_12b_it_tool, "gemma-3-12b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
tool_test!(test_gemma_3_27b_it_tool, "gemma-3-27b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);

// ── Gemma 3 (Image) ──────────────────────────────────────────────────────

/// Create a small test image (solid red 64x64 PNG) in memory.
fn create_test_image() -> Vec<u8> {
    let mut buf = Vec::new();
    let img = image::RgbImage::from_fn(64, 64, |_, _| image::Rgb([255u8, 0, 0]));
    let dyn_img = image::DynamicImage::ImageRgb8(img);
    let mut cursor = std::io::Cursor::new(&mut buf);
    dyn_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("failed to write test image");
    buf
}

macro_rules! image_test {
    ($name:ident, $model_name:expr, $model_type:path) => {
        image_test!($name, $model_name, $model_type, 64);
    };
    ($name:ident, $model_name:expr, $model_type:path, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let image_model = model.as_image().expect("should be an image model");
            let request = ImageAnalysisRequest {
                image_data: create_test_image(),
                prompt: "Describe this image briefly.".into(),
                max_tokens: $max_tokens,
                temperature: 0.1,
            };
            let response = image_model
                .analyze_image(&request)
                .expect("image analysis failed");

            eprintln!("image response: {:?}", response.text);
            assert!(!response.text.is_empty(), "image analysis returned empty text");
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

image_test!(test_gemma_3_4b_it_image, "gemma-3-4b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
image_test!(test_gemma_3_12b_it_image, "gemma-3-12b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);
image_test!(test_gemma_3_27b_it_image, "gemma-3-27b-it", nexo_ai::models::multipurpose::gemma3::Gemma3Model);

// ── Qwen3 (Chat) ───────────────────────────────────────────────────────────

chat_test!(test_qwen3_4b_q5km_chat, "qwen3-4b-q5km", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);
chat_test!(test_qwen3_30b_a3b_q4km_chat, "qwen3-30b-a3b-q4km", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);
chat_test!(test_qwen3_vl_4b_chat, "qwen3-vl-4b", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);

// ── Qwen3 (Tool) ───────────────────────────────────────────────────────────

tool_test!(test_qwen3_4b_q5km_tool, "qwen3-4b-q5km", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);
tool_test!(test_qwen3_30b_a3b_q4km_tool, "qwen3-30b-a3b-q4km", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);
tool_test!(test_qwen3_vl_4b_tool, "qwen3-vl-4b", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);

// ── Qwen3 (Image) ──────────────────────────────────────────────────────────

image_test!(test_qwen3_vl_4b_image, "qwen3-vl-4b", nexo_ai::models::multipurpose::qwen3::Qwen3Model, 256);

// ── Qwen3 (Performance) ────────────────────────────────────────────────────

perf_test!(test_qwen3_4b_q5km_perf, "qwen3-4b-q5km", 15.0, nexo_ai::models::multipurpose::qwen3::Qwen3Model);
perf_test!(test_qwen3_30b_a3b_q4km_perf, "qwen3-30b-a3b-q4km", 8.0, nexo_ai::models::multipurpose::qwen3::Qwen3Model);
perf_test!(test_qwen3_vl_4b_perf, "qwen3-vl-4b", 10.0, nexo_ai::models::multipurpose::qwen3::Qwen3Model);
