//! Integration tests for nexo-ai model inference.
//!
//! These tests load real models, run inference, and validate output.
//! They are `#[ignore]` by default since they require downloaded models and hardware.
//!
//! Run all:  `cargo test -p nexo-ai --test model_inference -- --ignored`
//! Run one:  `cargo test -p nexo-ai --test model_inference -- --ignored test_whisper_large_v3_turbo`

use std::path::Path;

use ntest::timeout;
use serial_test::serial;

use nexo_ai::download::paths::model_storage_dir;
use nexo_ai::registry::manifest::find_manifest;
use nexo_ai::shared::model_traits::ModelInfo;
use nexo_ai::shared::types::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Resolve model directory and memory estimate from the manifest registry.
/// Panics with a clear download instruction if the model is not downloaded.
fn resolve_model(model_name: &str) -> (std::path::PathBuf, u64) {
    let manifest = find_manifest(model_name)
        .unwrap_or_else(|| panic!("unknown model '{model_name}' in manifest registry"));

    let dir = model_storage_dir(model_name);
    if !dir.exists() {
        panic!(
            "Model '{}' not downloaded. Run: nexo-ai pull {}\n  (expected directory: {})",
            model_name,
            model_name,
            dir.display()
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

fn gemma3_chat_request() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "What is 2+2? Answer with just the number.".into(),
        }],
        max_tokens: 32,
        temperature: 0.1,
        top_p: 0.9,
    }
}

macro_rules! chat_test {
    ($name:ident, $model_name:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::multipurpose::gemma3::Gemma3Model::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            );

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");
            let response = chat
                .chat(&gemma3_chat_request())
                .expect("chat failed");

            eprintln!("response: {:?}", response.text);
            assert!(!response.text.is_empty(), "chat returned empty text");
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

chat_test!(test_gemma_3_4b_it_chat, "gemma-3-4b-it");
chat_test!(test_gemma_3_12b_it_chat, "gemma-3-12b-it");
chat_test!(test_gemma_3_27b_it_chat, "gemma-3-27b-it");

// ── Gemma 3 (Tool) ─────────────────────────────────────────────────────────

macro_rules! tool_test {
    ($name:ident, $model_name:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::multipurpose::gemma3::Gemma3Model::new(
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
                max_tokens: 128,
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

tool_test!(test_gemma_3_4b_it_tool, "gemma-3-4b-it");
tool_test!(test_gemma_3_12b_it_tool, "gemma-3-12b-it");
tool_test!(test_gemma_3_27b_it_tool, "gemma-3-27b-it");

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
    ($name:ident, $model_name:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::multipurpose::gemma3::Gemma3Model::new(
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
                max_tokens: 64,
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

image_test!(test_gemma_3_4b_it_image, "gemma-3-4b-it");
image_test!(test_gemma_3_12b_it_image, "gemma-3-12b-it");
image_test!(test_gemma_3_27b_it_image, "gemma-3-27b-it");
