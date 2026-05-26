#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(feature = "mlx")]

mod common;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use nexo_ai::api::model_traits::{ModelInfo, ToolModel};
use nexo_ai::api::types::{MessageRole, ModelCategory, ToolCallRequest, TranscriptMessage};
use nexo_ai::inference::models::gemma4::openai::Gemma4OpenAiFamily;
use nexo_ai::inference::remote::openai::model::{OpenAiModel, OpenAiServerControl};

#[derive(Clone, Default)]
struct NoopServer;

impl OpenAiServerControl for NoopServer {
    fn ensure_running(&self) -> Result<()> {
        Ok(())
    }

    fn unload_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }
}

fn tool_request(messages: Vec<TranscriptMessage>) -> ToolCallRequest {
    ToolCallRequest {
        messages,
        tools: vec![nexo_spec::tool::ToolSpec {
            name: "echo.run".into(),
            description: "Echo input".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            }),
            ..Default::default()
        }],
        max_tokens: 128,
        temperature: 0.1,
        top_p: 0.9,
        top_k: None,
        session_id: Some("session-1".into()),
    }
}

fn spawn_mock_openai_server(
    responses: Vec<serde_json::Value>,
) -> (
    String,
    Arc<Mutex<Vec<serde_json::Value>>>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
    let addr = listener.local_addr().expect("mock server local addr");
    let captured_requests = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::clone(&captured_requests);

    let handle = thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().expect("accept mock connection");
            let request = read_http_json(&mut stream);
            requests
                .lock()
                .expect("lock captured requests")
                .push(request);
            write_http_json(&mut stream, &response);
        }
    });

    (format!("http://{addr}"), captured_requests, handle)
}

fn read_http_json(stream: &mut TcpStream) -> serde_json::Value {
    let mut buffer = Vec::new();
    let mut content_length = None;
    let mut body_start = 0usize;

    loop {
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk).expect("read mock request");
        assert!(read > 0, "mock request closed before complete body");
        buffer.extend_from_slice(&chunk[..read]);

        if content_length.is_none()
            && let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
        {
            body_start = header_end + 4;
            let headers = std::str::from_utf8(&buffer[..header_end]).expect("utf8 headers");
            content_length = Some(parse_content_length(headers));
        }

        if let Some(length) = content_length
            && buffer.len() >= body_start + length
        {
            return serde_json::from_slice(&buffer[body_start..body_start + length])
                .expect("parse mock request json");
        }
    }
}

fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length").then(|| {
                value
                    .trim()
                    .parse::<usize>()
                    .expect("numeric content-length")
            })
        })
        .expect("content-length header")
}

fn write_http_json(stream: &mut TcpStream, payload: &serde_json::Value) {
    let body = serde_json::to_vec(payload).expect("serialize mock response");
    let headers = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .expect("write mock response headers");
    stream.write_all(&body).expect("write mock response body");
    stream.flush().expect("flush mock response");
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_followup_round_serializes_structured_assistant_history() {
    common::init_tracing();

    let (base_url, captured_requests, server_handle) = spawn_mock_openai_server(vec![
        serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "function": {
                            "name": "echo.run",
                            "arguments": "{\"message\":\"hello\"}"
                        }
                    }]
                }
            }],
            "usage": {"completion_tokens": 8}
        }),
        serde_json::json!({
            "choices": [{
                "message": {
                    "content": "The tool output contains the echoed value hello.",
                    "tool_calls": []
                }
            }],
            "usage": {"completion_tokens": 12}
        }),
    ]);

    let mut model = OpenAiModel::new(
        "mock-model",
        PathBuf::from("/tmp/mock-model"),
        1,
        vec![ModelCategory::Tool],
        Gemma4OpenAiFamily,
        NoopServer,
        &base_url,
    )
    .with_request_model_id("mock-model");
    model.load().expect("load mock model");

    let initial = model
        .call_tools(&tool_request(vec![TranscriptMessage::new(
            MessageRole::User,
            "say hello",
        )]))
        .expect("initial tool selection succeeds");
    assert_eq!(initial.tool_calls.len(), 1);
    assert_eq!(initial.tool_calls[0].name, "echo.run");

    let follow_up = model
        .call_tools(&tool_request(vec![
            TranscriptMessage::new(MessageRole::User, "say hello"),
            TranscriptMessage::new(
                MessageRole::Assistant,
                "<|tool_call>call:echo.run{message:<|\"|>hello<|\"|>}<tool_call|>",
            ),
            TranscriptMessage::with_tool_metadata(
                MessageRole::Tool,
                "exit_code: 0\nstdout:\nhello\n",
                Some("call-1".into()),
                Some("echo.run".into()),
            ),
        ]))
        .expect("follow-up tool round succeeds");

    assert!(follow_up.tool_calls.is_empty());
    assert_eq!(
        follow_up.reasoning.as_deref(),
        Some("The tool output contains the echoed value hello.")
    );

    server_handle.join().expect("join mock server");
    let requests = captured_requests
        .lock()
        .expect("lock captured requests for assertions");
    assert_eq!(requests.len(), 2);

    let second_request = &requests[1];
    let messages = second_request["messages"]
        .as_array()
        .expect("second request messages array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "assistant");
    assert!(messages[1]["content"].is_null());
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call-1");
    assert_eq!(messages[1]["tool_calls"][0]["type"], "function");
    assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "echo.run");
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["arguments"],
        "{\"message\":\"hello\"}"
    );
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call-1");
    assert_eq!(messages[2]["name"], "echo.run");
    assert_eq!(messages[2]["content"], "exit_code: 0\nstdout:\nhello\n");
}
