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

use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use ntest::timeout;
use serial_test::serial;

use nexo_ai::api::model_traits::ModelInfo;
use nexo_ai::api::types::{
    ChatMessage, ChatRequest, ChatRole, ImageAnalysisRequest, ModelCategory, ToolCallRequest,
};
use nexo_ai::models::gemma4::openai::{Gemma4OpenAiFamily, default_request_model_id};
use nexo_ai::openai::client::OpenAiClient;
use nexo_ai::openai::model::OpenAiModel;
use nexo_ai::openai::protocol::{OpenAiChatRequest, OpenAiContent, OpenAiMessage, OpenAiModelInfo};
use nexo_ai::servers::mlx_vlm::{MlxHealthInfo, MlxVlmHandle};

const DEFAULT_VENV_PATH: &str = "/Users/Mathijs.Mortimer/Development/utilities/.venv";
const SERVER_HOST: &str = "127.0.0.1";
// The local 8-bit manifest is the default MLX request target. Set
// `NEXO_AI_MLX_TEST_REQUEST_MODEL` to override this explicitly for experiments.
const TEST_MODEL: &str = "mlx-gemma-4-e2b-it-8bit";

struct TestRuntime {
    model_name: String,
    model_dir: PathBuf,
    request_model_id: String,
    memory_bytes: u64,
    base_url: String,
    server: MlxVlmHandle,
}

impl TestRuntime {
    fn from_env() -> Self {
        common::init_tracing();

        let host = env::var("NEXO_AI_MLX_TEST_HOST").unwrap_or_else(|_| SERVER_HOST.to_string());
        let port = env::var("NEXO_AI_MLX_TEST_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or_else(pick_unused_port);
        let model_name = env::var("NEXO_AI_MLX_TEST_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| TEST_MODEL.to_string());
        let (model_dir, memory_bytes) = common::resolve_model(&model_name);
        let default_model_id = default_request_model_id(&model_name, &model_dir);
        let request_model_id = env::var("NEXO_AI_MLX_TEST_REQUEST_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(default_model_id);
        let base_url = format!("http://{host}:{port}");
        let server = MlxVlmHandle::new(&host, port, resolve_venv_path());

        Self {
            model_name,
            model_dir,
            request_model_id,
            memory_bytes,
            base_url,
            server,
        }
    }

    fn client(&self) -> OpenAiClient {
        OpenAiClient::new(&self.base_url)
    }

    fn request_model_id(&self) -> &str {
        &self.request_model_id
    }

    fn create_remote_model(&self) -> OpenAiModel<Gemma4OpenAiFamily, MlxVlmHandle> {
        OpenAiModel::new(
            &self.model_name,
            self.model_dir.clone(),
            self.memory_bytes,
            vec![
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image,
            ],
            Gemma4OpenAiFamily,
            self.server.clone(),
            &self.base_url,
        )
        .with_request_model_id(self.request_model_id.clone())
    }

    async fn start_server(&self) -> Result<()> {
        self.server.shared().lock().await.start().await
    }

    async fn ensure_running(&self) -> Result<()> {
        self.server.shared().lock().await.ensure_running().await
    }

    async fn stop_server(&self) {
        self.server.shared().lock().await.stop().await;
    }

    async fn is_running(&self) -> bool {
        self.server.shared().lock().await.is_running()
    }

    async fn health_check(&self) -> Result<bool> {
        self.server.shared().lock().await.health_check().await
    }

    async fn health_info(&self) -> Result<MlxHealthInfo> {
        self.server.shared().lock().await.health_info().await
    }

    async fn list_models(&self) -> Result<Vec<OpenAiModelInfo>> {
        self.server.shared().lock().await.list_models().await
    }

    async fn unload_model(&self) -> Result<()> {
        self.server.shared().lock().await.unload_model().await
    }
}

fn resolve_venv_path() -> Option<String> {
    env::var("NEXO_AI_MLX_TEST_VENV")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            Path::new(DEFAULT_VENV_PATH)
                .exists()
                .then(|| DEFAULT_VENV_PATH.to_string())
        })
}

fn pick_unused_port() -> u16 {
    TcpListener::bind((SERVER_HOST, 0))
        .expect("failed to bind an ephemeral test port")
        .local_addr()
        .expect("failed to read bound test port")
        .port()
}

fn text_request(model_id: &str, prompt: &str, max_tokens: usize) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model_id.to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: Some(OpenAiContent::Text(prompt.to_string())),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }],
        max_tokens,
        temperature: 0.1,
        top_p: Some(0.9),
        tools: None,
    }
}

fn contains_any(text: &str, expected: &[&str]) -> bool {
    let normalized = text.to_ascii_lowercase();
    expected
        .iter()
        .any(|needle| normalized.contains(&needle.to_ascii_lowercase()))
}

fn ensure_model_metadata(models: &[OpenAiModelInfo]) -> Result<()> {
    ensure!(
        !models.is_empty(),
        "expected /v1/models to return at least one model"
    );
    for model in models {
        ensure!(
            !model.id.is_empty(),
            "model id should not be empty: {model:?}"
        );
        ensure!(
            !model.object.is_empty(),
            "model object should not be empty: {model:?}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_server_start_stop() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        ensure!(!runtime.is_running().await, "server should start stopped");
        runtime
            .start_server()
            .await
            .context("failed to start server")?;
        ensure!(
            runtime.is_running().await,
            "server should be running after start"
        );

        let healthy = runtime
            .health_check()
            .await
            .context("health check failed")?;
        ensure!(healthy, "server should be healthy after start");
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    assert!(!runtime.is_running().await, "server should stop cleanly");
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_list_models_after_loading_model() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        runtime
            .start_server()
            .await
            .context("failed to start server")?;

        runtime
            .client()
            .chat_completion(&text_request(
                runtime.request_model_id(),
                "Reply with OK.",
                8,
            ))
            .await
            .context("initial chat completion failed")?;

        let models = runtime.list_models().await.context("list_models failed")?;
        eprintln!("models after load: {models:?}");
        ensure_model_metadata(&models)?;

        let health = runtime.health_info().await.context("health info failed")?;
        ensure!(
            health.loaded_model.as_deref() == Some(runtime.request_model_id()),
            "expected /health to report the loaded local model path, got {:?}",
            health.loaded_model
        );
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_chat_inference() {
    let runtime = TestRuntime::from_env();
    let mut model = runtime.create_remote_model();

    let result: Result<()> = (|| {
        model.load().context("failed to load remote model")?;
        ensure!(model.is_loaded(), "remote model should report loaded");

        let response = model
            .as_chat()
            .context("expected MLX model to expose chat")?
            .chat(&ChatRequest {
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "What is 2+2? Answer with just the number.",
                )],
                max_tokens: 32,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
            })
            .context("remote chat failed")?;

        eprintln!("chat response: {}", response.text);
        ensure!(
            contains_any(&response.text, &["4", "four"]),
            "expected answer to mention 4, got: {}",
            response.text
        );
        Ok(())
    })();

    model.unload();
    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_tool_calling() {
    let runtime = TestRuntime::from_env();
    let mut model = runtime.create_remote_model();

    let result: Result<()> = (|| {
        model.load().context("failed to load remote model")?;

        let response = model
            .as_tool()
            .context("expected MLX model to expose tool calling")?
            .call_tools(&ToolCallRequest {
                messages: vec![ChatMessage::new(
                    ChatRole::User,
                    "What is the weather in Amsterdam? Use the weather tool.",
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
                max_tokens: 96,
                temperature: 0.1,
                top_p: 0.9,
                top_k: None,
                session_id: None,
            })
            .context("remote tool call failed")?;

        eprintln!("tool calls: {:?}", response.tool_calls);
        eprintln!("reasoning: {:?}", response.reasoning);

        ensure!(
            !response.tool_calls.is_empty(),
            "expected at least one parsed tool call, got {:?}",
            response.reasoning
        );

        let first = &response.tool_calls[0];
        ensure!(
            first.name == "get_weather",
            "unexpected tool name: {}",
            first.name
        );
        let city = first
            .arguments
            .get("city")
            .and_then(|value| value.as_str())
            .context("expected parsed tool arguments to contain a city string")?;
        ensure!(
            city.to_ascii_lowercase().contains("amsterdam"),
            "expected tool arguments to mention Amsterdam, got: {city}"
        );
        Ok(())
    })();

    model.unload();
    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_image_analysis() {
    let runtime = TestRuntime::from_env();
    let mut model = runtime.create_remote_model();

    let result: Result<()> = (|| {
        model.load().context("failed to load remote model")?;

        let response = model
            .as_image()
            .context("expected MLX model to expose image analysis")?
            .analyze_image(&ImageAnalysisRequest {
                image_data: common::create_test_png(),
                prompt: "What is the dominant color in this image? Answer with one word.".into(),
                max_tokens: 16,
                temperature: 0.1,
            })
            .context("remote image analysis failed")?;

        eprintln!("image analysis response: {}", response.text);
        ensure!(
            contains_any(&response.text, &["red"]),
            "expected image analysis to identify red, got: {}",
            response.text
        );
        Ok(())
    })();

    model.unload();
    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_unload_model() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        runtime
            .start_server()
            .await
            .context("failed to start server")?;
        let client = runtime.client();

        let first = client
            .chat_completion(&text_request(
                runtime.request_model_id(),
                "Reply with hi.",
                8,
            ))
            .await
            .context("initial chat failed")?;
        let first_text = first
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .context("first response had no content")?;
        ensure!(
            !first_text.trim().is_empty(),
            "first response should not be empty"
        );

        runtime.unload_model().await.context("unload failed")?;

        let second = client
            .chat_completion(&text_request(
                runtime.request_model_id(),
                "Reply with hi again.",
                8,
            ))
            .await
            .context("reload chat failed")?;
        let second_text = second
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .context("second response had no content")?;
        ensure!(
            !second_text.trim().is_empty(),
            "second response should not be empty"
        );
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(300_000)]
async fn test_sequential_requests() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        runtime
            .start_server()
            .await
            .context("failed to start server")?;
        let client = runtime.client();

        for i in 0..3 {
            let response = client
                .chat_completion(&text_request(
                    runtime.request_model_id(),
                    &format!("Count from 1 to {}. Be brief.", i + 2),
                    24,
                ))
                .await
                .with_context(|| format!("request {i} failed"))?;

            let text = response
                .choices
                .first()
                .and_then(|choice| choice.message.content.as_ref())
                .with_context(|| format!("request {i} returned no content"))?;

            eprintln!("response {i}: {text}");
            ensure!(
                !text.trim().is_empty(),
                "request {i} returned an empty response"
            );
        }

        ensure!(
            runtime.is_running().await,
            "server should still be running after sequential requests"
        );
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_ensure_running_starts_server() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        ensure!(!runtime.is_running().await, "server should start stopped");
        runtime
            .ensure_running()
            .await
            .context("ensure_running should start the server")?;
        ensure!(runtime.is_running().await, "server should be running");
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
#[timeout(120_000)]
async fn test_ensure_running_idempotent() {
    let runtime = TestRuntime::from_env();

    let result: Result<()> = async {
        runtime
            .start_server()
            .await
            .context("failed to start server")?;
        ensure!(runtime.is_running().await, "server should be running");

        runtime
            .ensure_running()
            .await
            .context("ensure_running failed on running server")?;
        ensure!(runtime.is_running().await, "server should still be running");
        Ok(())
    }
    .await;

    runtime.stop_server().await;
    result.unwrap();
}
