//! Integration tests for the MLX VLM server backend.
//!
//! These tests start the real `mlx_vlm.server` Python process, send requests,
//! and validate responses. They are `#[ignore]` by default since they require:
//! - A Python venv with `mlx-vlm` installed
//! - Downloaded MLX model files
//! - macOS with Apple Silicon
//!
//! Run all:  `cargo test -p nexo-ai --test mlx_server -- --ignored`
//! Run one:  `cargo test -p nexo-ai --test mlx_server -- --ignored test_server_start_stop`
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "mlx")]

mod common;

use ntest::timeout;
use serial_test::serial;

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use nexo_ai::remote_models::mlx_server::MlxServer;
use nexo_ai::remote_models::openai_client::{
    ImageUrlDetail, OpenAiChatRequest, OpenAiClient, OpenAiContent, OpenAiContentPart,
    OpenAiMessage,
};

// ── Constants ──────────────────────────────────────────────────────────────

const VENV_PATH: &str = "/Users/Mathijs.Mortimer/Development/utilities/.venv";
const SERVER_HOST: &str = "127.0.0.1";
const SERVER_PORT: u16 = 8089;
const TEST_MODEL: &str = "mlx-gemma-4-e2b-it-8bit";

// ── Helpers ────────────────────────────────────────────────────────────────

fn create_server() -> MlxServer {
    MlxServer::new(SERVER_HOST, SERVER_PORT, Some(VENV_PATH.to_string()))
}

fn create_client() -> OpenAiClient {
    OpenAiClient::new(&format!("http://{SERVER_HOST}:{SERVER_PORT}"))
}

fn resolve_model_dir(model_name: &str) -> std::path::PathBuf {
    common::resolve_model(model_name).0
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
#[serial]
#[timeout(60_000)]
async fn test_server_start_stop() {
    common::init_tracing();

    let mut server = create_server();
    assert!(!server.is_running());

    server.start().await.expect("failed to start server");
    assert!(server.is_running());

    let healthy = server.health_check().await.expect("health check failed");
    assert!(healthy, "server should be healthy after start");

    server.stop().await;
    assert!(!server.is_running());
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(60_000)]
async fn test_list_models() {
    common::init_tracing();

    let mut server = create_server();
    server.start().await.expect("failed to start server");

    let models = server.list_models().await.expect("list_models failed");
    eprintln!("models: {models:?}");

    server.stop().await;
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_chat_inference() {
    common::init_tracing();
    let model_dir = resolve_model_dir(TEST_MODEL);

    let mut server = create_server();
    server.start().await.expect("failed to start server");

    let client = create_client();
    let request = OpenAiChatRequest {
        model: model_dir.to_string_lossy().to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Text("What is 2+2? Answer with just the number.".to_string()),
        }],
        max_tokens: 32,
        temperature: 0.1,
        top_p: Some(0.9),
    };

    let response = client
        .chat_completion(&request)
        .await
        .expect("chat completion failed");

    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .expect("no content in response");

    eprintln!("response: {text}");
    assert!(!text.is_empty(), "response should not be empty");

    if let Some(usage) = &response.usage {
        eprintln!("usage: completion_tokens={:?}", usage.completion_tokens);
    }

    server.stop().await;
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_image_analysis() {
    common::init_tracing();
    let model_dir = resolve_model_dir(TEST_MODEL);

    let mut server = create_server();
    server.start().await.expect("failed to start server");

    let test_image = common::create_test_png();
    let b64 = B64.encode(&test_image);
    let data_uri = format!("data:image/png;base64,{b64}");

    let client = create_client();
    let request = OpenAiChatRequest {
        model: model_dir.to_string_lossy().to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Parts(vec![
                OpenAiContentPart::ImageUrl {
                    image_url: ImageUrlDetail { url: data_uri },
                },
                OpenAiContentPart::Text {
                    text: "Describe this image briefly.".to_string(),
                },
            ]),
        }],
        max_tokens: 64,
        temperature: 0.1,
        top_p: None,
    };

    let response = client
        .chat_completion(&request)
        .await
        .expect("image chat failed");

    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .expect("no content in response");

    eprintln!("image analysis response: {text}");
    assert!(!text.is_empty(), "response should not be empty");

    server.stop().await;
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_unload_model() {
    common::init_tracing();
    let model_dir = resolve_model_dir(TEST_MODEL);

    let mut server = create_server();
    server.start().await.expect("failed to start server");

    let client = create_client();
    let request = OpenAiChatRequest {
        model: model_dir.to_string_lossy().to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: OpenAiContent::Text("Hi".to_string()),
        }],
        max_tokens: 4,
        temperature: 0.1,
        top_p: None,
    };
    client
        .chat_completion(&request)
        .await
        .expect("initial chat failed");

    server.unload_model().await.expect("unload failed");

    let models = server.list_models().await.expect("list_models failed");
    eprintln!("models after unload: {models:?}");

    server.stop().await;
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_sequential_requests() {
    common::init_tracing();
    let model_dir = resolve_model_dir(TEST_MODEL);

    let mut server = create_server();
    server.start().await.expect("failed to start server");

    let client = create_client();
    let model_id = model_dir.to_string_lossy().to_string();

    for i in 0..3 {
        let request = OpenAiChatRequest {
            model: model_id.clone(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: OpenAiContent::Text(format!("Count to {i}. Be brief.")),
            }],
            max_tokens: 32,
            temperature: 0.1,
            top_p: None,
        };

        let response = client
            .chat_completion(&request)
            .await
            .unwrap_or_else(|e| panic!("request {i} failed: {e}"));

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .expect("no content");

        eprintln!("response {i}: {text}");
        assert!(!text.is_empty());
    }

    assert!(server.is_running(), "server should still be running");
    server.stop().await;
}

#[tokio::test]
#[ignore]
#[serial]
#[timeout(60_000)]
async fn test_ensure_running_idempotent() {
    common::init_tracing();

    let mut server = create_server();
    server.start().await.expect("failed to start server");
    assert!(server.is_running());

    server
        .ensure_running()
        .await
        .expect("ensure_running failed on running server");
    assert!(server.is_running());

    server.stop().await;
}
