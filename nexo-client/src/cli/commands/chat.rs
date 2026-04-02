use super::helpers::{connect_and_handshake, recv_response};
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::{
    AgentEventPayload, AgentParams, AgentResponse, AgentStatus, EventKind, Frame, Method,
    SessionCreateParams, SessionCreateResponse,
};
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};

pub struct ChatOptions {
    pub url_override: Option<String>,
    pub session_id: Option<String>,
    pub session_name: Option<String>,
    pub model_id: Option<String>,
}

pub async fn run_chat(opts: ChatOptions) -> utl_helpers::Result {
    let (mut conn, _hello) = connect_and_handshake(opts.url_override.as_deref()).await?;

    let session_id = match opts.session_id {
        Some(id) => {
            println!("Resuming session {id}");
            id
        }
        None => {
            let name = opts.session_name.unwrap_or_else(|| "cli-chat".to_string());
            let session = create_session(&mut conn, &name).await?;
            println!("Created session {} ({})", session.session_id, name);
            session.session_id
        }
    };

    println!("Type your message or 'quit' to exit.\n");

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        print!("you> ");
        std::io::stdout().flush().ok();

        let prompt = match lines.next_line().await {
            Ok(Some(line)) => {
                let line = line.trim().to_string();
                if line == "quit" || line == "exit" {
                    break;
                }
                if line.is_empty() {
                    continue;
                }
                line
            }
            Ok(None) => break,
            Err(e) => {
                tracing::error!("Stdin error: {e}");
                break;
            }
        };

        let run_id = match send_agent(&mut conn, &prompt, &session_id, &opts.model_id).await {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Error: {e}");
                continue;
            }
        };

        if let Err(e) = stream_response(&mut conn, &run_id).await {
            eprintln!("Error streaming response: {e}");
        }
    }

    if let Err(e) = conn.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }
    println!("\nDisconnected.");
    Ok(())
}

async fn create_session(
    conn: &mut NexoConnection,
    name: &str,
) -> utl_helpers::Result<SessionCreateResponse> {
    let frame = Frame::request(
        Method::SessionCreate,
        &SessionCreateParams {
            name: Some(name.to_string()),
            prefill_collection_id: None,
        },
    )
    .map_err(|e| utl_helpers::Error::Network(format!("Failed to build request: {e}")))?;

    conn.send_frame(&frame)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send failed: {e}")))?;

    let response = recv_response(conn).await?;
    let payload: SessionCreateResponse = serde_json::from_value(response)
        .map_err(|e| utl_helpers::Error::Network(format!("Invalid session response: {e}")))?;

    Ok(payload)
}

async fn send_agent(
    conn: &mut NexoConnection,
    prompt: &str,
    session_id: &str,
    model_id: &Option<String>,
) -> std::result::Result<String, String> {
    let params = AgentParams {
        prompt: prompt.to_string(),
        idempotency_key: Frame::new_id(),
        session_id: Some(session_id.to_string()),
        context: None,
        model_id: model_id.clone(),
    };

    let frame =
        Frame::request(Method::Agent, &params).map_err(|e| format!("Build request: {e}"))?;

    conn.send_frame(&frame)
        .await
        .map_err(|e| format!("Send failed: {e}"))?;

    let response = recv_response(conn)
        .await
        .map_err(|e| format!("Response: {e}"))?;

    let agent_resp: AgentResponse = serde_json::from_value(response)
        .map_err(|e| format!("Invalid agent response: {e}"))?;

    tracing::debug!("Agent run {} status: {:?}", agent_resp.run_id, agent_resp.status);

    Ok(agent_resp.run_id)
}

/// Receive frames until an agent event with status completed/failed is seen for `run_id`.
async fn stream_response(
    conn: &mut NexoConnection,
    run_id: &str,
) -> std::result::Result<(), String> {
    let mut prev_len = 0usize;

    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|e| format!("Receive error: {e}"))?
            .ok_or("Connection closed while waiting for response")?;

        match frame {
            Frame::Event {
                event, payload, ..
            } if event == EventKind::Agent => {
                let agent_event: AgentEventPayload =
                    serde_json::from_value(payload).map_err(|e| format!("Parse: {e}"))?;

                if agent_event.run_id != run_id {
                    continue;
                }

                match agent_event.status {
                    AgentStatus::Thinking => {
                        eprint!("  [thinking]");
                        std::io::stderr().flush().ok();
                    }
                    AgentStatus::Queued => {
                        eprint!("  [queued - waiting for inference node]");
                        std::io::stderr().flush().ok();
                    }
                    AgentStatus::ToolCall => {
                        let tool = agent_event.tool_name.as_deref().unwrap_or("unknown");
                        eprint!("\r\x1b[K  [tool: {tool}]");
                        std::io::stderr().flush().ok();
                    }
                    AgentStatus::Streaming => {
                        if let Some(content) = &agent_event.content {
                            if prev_len == 0 {
                                eprint!("\r\x1b[K");
                                print!("assistant> ");
                            }
                            // Gateway sends cumulative content — print only the delta
                            print!("{}", &content[prev_len..]);
                            std::io::stdout().flush().ok();
                            prev_len = content.len();
                        }
                    }
                    AgentStatus::Completed => {
                        if let Some(content) = &agent_event.content {
                            if prev_len == 0 {
                                eprint!("\r\x1b[K");
                                print!("assistant> ");
                            }
                            print!("{}", &content[prev_len..]);
                        }
                        println!();
                        println!();
                        return Ok(());
                    }
                    AgentStatus::Failed => {
                        if prev_len > 0 {
                            println!();
                        } else {
                            eprint!("\r\x1b[K");
                        }
                        let error =
                            agent_event.error.or(agent_event.content).unwrap_or_default();
                        eprintln!("Error: {error}");
                        println!();
                        return Ok(());
                    }
                    AgentStatus::Accepted | AgentStatus::Cancelled => {}
                }
            }
            Frame::Event { event, .. } if event == EventKind::Tick => {}
            _ => {
                tracing::debug!("Unexpected frame during chat: {frame:?}");
            }
        }
    }
}
