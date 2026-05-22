//! Integration tests for nexo-ai model inference.
//!
//! These tests load real models, run inference, and validate output.
//! They are `#[ignore]` by default since they require downloaded models and hardware.
//!
//! Run all:  `cargo test -p nexo-ai --test model_inference -- --ignored`
//! Run one:  `cargo test -p nexo-ai --test model_inference -- --ignored test_whisper_large_v3_turbo`
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::path::Path;

use ntest::timeout;
use serial_test::serial;

use common::resolve_model;
use nexo_ai::api::model_traits::ModelInfo;
use nexo_ai::api::types::*;

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
            let mut model = nexo_ai::models::whisper::WhisperModel::new(
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

// ── Imagine ─────────────────────────────────────────────────────────────────

macro_rules! imagine_test {
    ($name:ident, $model_name:expr, $model_type:path) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new($model_name.into(), memory_bytes, model_dir);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let imagine = model.as_imagine().expect("should be an imagine model");
            let request = ImagineRequest {
                prompt: "a red circle on white background".into(),
                width: 256,
                height: 256,
                steps: 1,
                guidance: 3.5,
                seed: 42,
                batch_size: 1,
            };
            let response = imagine.imagine(&request).expect("image generation failed");

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

imagine_test!(
    test_flux_2_klein_4b,
    "flux-2-klein-4b",
    nexo_ai::models::flux2::FluxModel
);
imagine_test!(
    test_flux_2_klein_9b,
    "flux-2-klein-9b",
    nexo_ai::models::flux2::FluxModel
);
imagine_test!(
    test_flux_2_dev,
    "flux-2-dev",
    nexo_ai::models::flux2::FluxModel
);
imagine_test!(
    test_z_image_turbo,
    "z-image-turbo",
    nexo_ai::models::z_image::ZImageModel
);
imagine_test!(
    test_qwen_image_q4,
    "qwen-image-q4",
    nexo_ai::models::qwen_image::QwenImageModel
);
imagine_test!(
    test_qwen_image_q6,
    "qwen-image-q6",
    nexo_ai::models::qwen_image::QwenImageModel
);
imagine_test!(
    test_qwen_image_q8,
    "qwen-image-q8",
    nexo_ai::models::qwen_image::QwenImageModel
);
imagine_test!(
    test_qwen_image_bf16,
    "qwen-image-bf16",
    nexo_ai::models::qwen_image::QwenImageModel
);

// ── Imagine (File Output) ───────────────────────────────────────────────────

/// Generate an image, save to a file, and verify the PNG is valid and non-trivial.
macro_rules! imagine_file_test {
    ($name:ident, $model_name:expr, $model_type:path, $prompt:expr, $filename:expr, $steps:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(900_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new($model_name.into(), memory_bytes, model_dir);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            // Use a closure so model.unload() runs even on assertion failure.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let imagine = model.as_imagine().expect("should be an imagine model");
                let request = ImagineRequest {
                    prompt: $prompt.into(),
                    width: 256,
                    height: 256,
                    steps: $steps,
                    guidance: 3.5,
                    seed: 42,
                    batch_size: 1,
                };
                let response = imagine.imagine(&request).expect("image generation failed");

                assert!(!response.images.is_empty(), "no images generated");
                let img = &response.images[0];
                assert!(
                    img.data.len() > 1000,
                    "image data suspiciously small ({} bytes)",
                    img.data.len()
                );
                assert_eq!(img.width, 256);
                assert_eq!(img.height, 256);

                // Verify valid PNG by decoding
                let decoded =
                    image::load_from_memory_with_format(&img.data, image::ImageFormat::Png)
                        .expect("generated data is not valid PNG");
                assert_eq!(decoded.width(), 256);
                assert_eq!(decoded.height(), 256);

                // Save to file first (before assertions that might fail)
                let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("datasets/images/generated");
                std::fs::create_dir_all(&out_dir).expect("failed to create output directory");
                let out_path = out_dir.join($filename);
                std::fs::write(&out_path, &img.data).expect("failed to write image");
                eprintln!(
                    "saved {} ({} bytes, {}x{}, {}ms)",
                    out_path.display(),
                    img.data.len(),
                    img.width,
                    img.height,
                    response.inference_time_ms,
                );

                // Check image isn't a solid color (garbled/blank)
                let rgb = decoded.to_rgb8();
                let first_pixel = rgb.get_pixel(0, 0);
                let center_pixel = rgb.get_pixel(128, 128);
                let has_variation =
                    first_pixel != center_pixel || rgb.pixels().any(|p| p != first_pixel);
                assert!(
                    has_variation,
                    "image appears to be a single solid color (likely garbled)"
                );
            }));

            model.unload();
            assert!(!model.is_loaded());

            if let Err(panic) = result {
                std::panic::resume_unwind(panic);
            }
        }
    };
}

imagine_file_test!(
    test_z_image_turbo_avocado,
    "z-image-turbo",
    nexo_ai::models::z_image::ZImageModel,
    "avocado",
    "z-image-test-avocado.png",
    4
);

imagine_file_test!(
    test_flux_2_klein_4b_avocado,
    "flux-2-klein-4b",
    nexo_ai::models::flux2::FluxModel,
    "avocado",
    "flux-test-avocado.png",
    4
);

// ── Gemma 4 (Chat) ─────────────────────────────────────────────────────────

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
            let mut model = <$model_type>::new($model_name.into(), memory_bytes, model_dir);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");
            let request = ChatRequest {
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "What is 2+2? Answer with just the number.",
                )],
                max_tokens: $max_tokens,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
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

chat_test!(
    test_gemma_4_e4b_chat,
    "gemma-4-e4b",
    nexo_ai::models::gemma4::Gemma4Model
);
chat_test!(
    test_gemma_4_e2b_it_chat,
    "gemma-4-e2b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
chat_test!(
    test_gemma_4_e4b_it_chat,
    "gemma-4-e4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
chat_test!(
    test_gemma_4_26b_a4b_it_chat,
    "gemma-4-26b-a4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
chat_test!(
    test_gemma_4_31b_it_chat,
    "gemma-4-31b-it",
    nexo_ai::models::gemma4::Gemma4Model
);

// ── Gemma 4 (Performance) ──────────────────────────────────────────────────

macro_rules! perf_test {
    ($name:ident, $model_name:expr, $min_tok_per_sec:expr, $model_type:path) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = <$model_type>::new($model_name.into(), memory_bytes, model_dir);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");

            // Warmup: prime Metal shader compilation
            let warmup = ChatRequest {
                messages: vec![ChatMessage::new(ChatRole::User, "Hi")],
                max_tokens: 2,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
            };
            let _ = chat.chat(&warmup);

            // Benchmark
            let request = ChatRequest {
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "Write a short paragraph about the history of computing.",
                )],
                max_tokens: 128,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
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

perf_test!(
    test_gemma_4_e2b_it_perf,
    "gemma-4-e2b-it",
    15.0,
    nexo_ai::models::gemma4::Gemma4Model
);
perf_test!(
    test_gemma_4_e4b_it_perf,
    "gemma-4-e4b-it",
    10.0,
    nexo_ai::models::gemma4::Gemma4Model
);
perf_test!(
    test_gemma_4_26b_a4b_it_perf,
    "gemma-4-26b-a4b-it",
    5.0,
    nexo_ai::models::gemma4::Gemma4Model
);
perf_test!(
    test_gemma_4_31b_it_perf,
    "gemma-4-31b-it",
    2.0,
    nexo_ai::models::gemma4::Gemma4Model
);

// ── Gemma 4 (Tool) ─────────────────────────────────────────────────────────

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
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "What is the weather in Amsterdam?",
                )],
                tools: vec![nexo_spec::tool::ToolSpec {
                    name: "get_weather".into(),
                    description: "Get the current weather for a city".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    }),
                    ..Default::default()
                }],
                max_tokens: $max_tokens,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
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

tool_test!(
    test_gemma_4_e2b_it_tool,
    "gemma-4-e2b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
tool_test!(
    test_gemma_4_e4b_it_tool,
    "gemma-4-e4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
tool_test!(
    test_gemma_4_26b_a4b_it_tool,
    "gemma-4-26b-a4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
tool_test!(
    test_gemma_4_31b_it_tool,
    "gemma-4-31b-it",
    nexo_ai::models::gemma4::Gemma4Model
);

// ── Gemma 4 GGUF (Chat) ───────────────────────────────────────────────────

macro_rules! gguf_chat_test {
    ($name:ident, $model_name:expr) => {
        gguf_chat_test!($name, $model_name, 32);
    };
    ($name:ident, $model_name:expr, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::gemma4::Gemma4Model::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            )
            .with_gguf(true);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let chat = model.as_chat().expect("should be a chat model");
            let request = ChatRequest {
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "What is 2+2? Answer with just the number.",
                )],
                max_tokens: $max_tokens,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
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

gguf_chat_test!(test_gemma_4_e2b_it_q5_chat, "gemma-4-e2b-it-q5");

// ── Gemma 4 GGUF (Image) ──────────────────────────────────────────────────

/// Load a test image from the datasets directory.
fn load_test_image_file(filename: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("datasets/images")
        .join(filename);
    assert!(path.exists(), "test image not found at {}", path.display());
    std::fs::read(&path).expect("failed to read test image")
}

macro_rules! gguf_image_test {
    ($name:ident, $model_name:expr, $image_file:expr, $prompt:expr) => {
        gguf_image_test!($name, $model_name, $image_file, $prompt, 128);
    };
    ($name:ident, $model_name:expr, $image_file:expr, $prompt:expr, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::gemma4::Gemma4Model::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            )
            .with_gguf(true);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let image_model = model.as_image().expect("should be an image model");
            let request = ImageAnalysisRequest {
                image_data: load_test_image_file($image_file),
                prompt: $prompt.into(),
                max_tokens: $max_tokens,
                temperature: 0.1,
            };
            let response = image_model
                .analyze_image(&request)
                .expect("image analysis failed");

            eprintln!("GGUF image response: {:?}", response.text);
            assert!(
                !response.text.is_empty(),
                "GGUF image analysis returned empty text"
            );
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

gguf_image_test!(
    test_gemma_4_e2b_it_q5_image,
    "gemma-4-e2b-it-q5",
    "mk2_pants_down.png",
    "What do you see in this image? Describe it briefly."
);

// ── Gemma 4 GGUF (Audio) ──────────────────────────────────────────────────

macro_rules! gguf_audio_test {
    ($name:ident, $model_name:expr) => {
        gguf_audio_test!($name, $model_name, 128);
    };
    ($name:ident, $model_name:expr, $max_tokens:expr) => {
        #[test]
        #[ignore]
        #[serial]
        #[timeout(600_000)]
        fn $name() {
            let (model_dir, memory_bytes) = resolve_model($model_name);
            let mut model = nexo_ai::models::gemma4::Gemma4Model::new(
                $model_name.into(),
                memory_bytes,
                model_dir,
            )
            .with_gguf(true);

            model.load().expect("failed to load model");
            assert!(model.is_loaded());

            let audio_model = model
                .as_audio_analysis()
                .expect("should be an audio analysis model");
            let (samples, sample_rate) = load_test_audio();
            let request = AudioAnalysisRequest {
                pcm_samples: samples,
                sample_rate,
                prompt: "What is being said in this audio?".into(),
                max_tokens: $max_tokens,
                temperature: 0.1,
            };
            let response = audio_model
                .analyze_audio(&request)
                .expect("audio analysis failed");

            eprintln!("GGUF audio response: {:?}", response.text);
            assert!(
                !response.text.is_empty(),
                "GGUF audio analysis returned empty text"
            );
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

gguf_audio_test!(test_gemma_4_e2b_it_q5_audio, "gemma-4-e2b-it-q5");

// ── Gemma 4 (Image) ──────────────────────────────────────────────────────

fn create_test_image() -> Vec<u8> {
    common::create_test_png()
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
            let mut model = <$model_type>::new($model_name.into(), memory_bytes, model_dir);

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
            assert!(
                !response.text.is_empty(),
                "image analysis returned empty text"
            );
            assert!(response.tokens_generated > 0);

            model.unload();
            assert!(!model.is_loaded());
        }
    };
}

image_test!(
    test_gemma_4_e2b_it_image,
    "gemma-4-e2b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
image_test!(
    test_gemma_4_e4b_it_image,
    "gemma-4-e4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
image_test!(
    test_gemma_4_26b_a4b_it_image,
    "gemma-4-26b-a4b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
image_test!(
    test_gemma_4_31b_it_image,
    "gemma-4-31b-it",
    nexo_ai::models::gemma4::Gemma4Model
);
