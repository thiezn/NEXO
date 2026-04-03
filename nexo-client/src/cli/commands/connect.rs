use super::helpers::connect_and_handshake;
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::{
    AgentParams, CronCreateParams, CronDeleteParams, Frame, Method, PrefillCollectionCreateParams,
    PrefillCollectionDeleteParams, PrefillMarkdownCreateParams, PrefillMarkdownDeleteParams,
    SessionClearParams, SessionCreateParams, SessionGetParams, SystemPresenceParams,
    ToolsExecuteParams,
};
use tokio::io::{AsyncBufReadExt, BufReader};

pub async fn run_connect(url_override: Option<String>) -> utl_helpers::Result {
    let (mut conn, _hello) = connect_and_handshake(url_override.as_deref()).await?;

    println!("Connected!");
    print_help();

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        tokio::select! {
            line = lines.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        let line = line.trim().to_string();
                        if line == "quit" || line == "exit" {
                            break;
                        }
                        if line.is_empty() {
                            continue;
                        }
                        if line == "help" {
                            print_help();
                            continue;
                        }
                        if let Err(e) = handle_user_input(&mut conn, &line).await {
                            tracing::error!("Send error: {e}");
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("Stdin error: {e}");
                        break;
                    }
                }
            }
            frame = conn.recv_frame() => {
                match frame {
                    Ok(Some(frame)) => {
                        match serde_json::to_string_pretty(&frame) {
                            Ok(json) => println!("<<< {json}"),
                            Err(e) => tracing::error!("Failed to format frame: {e}"),
                        }
                    }
                    Ok(None) => {
                        tracing::info!("Connection closed by server");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("Receive error: {e}");
                        break;
                    }
                }
            }
        }
    }

    if let Err(e) = conn.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }
    println!("Disconnected.");
    Ok(())
}

async fn handle_user_input(
    conn: &mut NexoConnection,
    input: &str,
) -> std::result::Result<(), String> {
    let (cmd, args) = input
        .split_once(' ')
        .map(|(c, a)| (c, a.trim()))
        .unwrap_or((input, ""));

    let frame = match cmd {
        "health" => request(Method::Health, &serde_json::json!({}))?,
        "status" => request(Method::Status, &serde_json::json!({}))?,
        "tools.catalog" => request(Method::ToolsCatalog, &serde_json::json!({}))?,
        "session.list" => request(Method::SessionList, &serde_json::json!({}))?,
        "cron.list" => request(Method::CronList, &serde_json::json!({}))?,
        "prefill.markdown.list" => request(Method::PrefillMarkdownList, &serde_json::json!({}))?,
        "prefill.collection.list" => {
            request(Method::PrefillCollectionList, &serde_json::json!({}))?
        }

        "system-presence" => request(
            Method::SystemPresence,
            &SystemPresenceParams {
                status: if args.is_empty() {
                    "active".to_string()
                } else {
                    args.to_string()
                },
            },
        )?,

        "agent" => {
            if args.is_empty() {
                println!("Usage: agent <prompt>");
                println!("       agent --session <id> <prompt>");
                println!("       agent --model <model_id> <prompt>");
                return Ok(());
            }
            let (session_id, model_id, prompt) = parse_agent_args(args);
            request(
                Method::Agent,
                &AgentParams {
                    prompt,
                    idempotency_key: uuid_v7(),
                    session_id,
                    context: None,
                    model_id,
                    thinking: None,
                },
            )?
        }

        "session.create" => {
            let name = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            request(
                Method::SessionCreate,
                &SessionCreateParams {
                    name,
                    prefill_collection_id: None,
                },
            )?
        }
        "session.get" => {
            if args.is_empty() {
                println!("Usage: session.get <sessionId>");
                return Ok(());
            }
            request(
                Method::SessionGet,
                &SessionGetParams {
                    session_id: args.to_string(),
                },
            )?
        }
        "session.clear" => {
            if args.is_empty() {
                println!("Usage: session.clear <sessionId>");
                return Ok(());
            }
            request(
                Method::SessionClear,
                &SessionClearParams {
                    session_id: args.to_string(),
                },
            )?
        }

        "cron.create" => {
            let params: CronCreateParams = parse_json_args(args, "cron.create <json>")?;
            request(Method::CronCreate, &params)?
        }
        "cron.delete" => {
            if args.is_empty() {
                println!("Usage: cron.delete <jobId>");
                return Ok(());
            }
            request(
                Method::CronDelete,
                &CronDeleteParams {
                    job_id: args.to_string(),
                },
            )?
        }

        "tools.execute" => {
            let params: ToolsExecuteParams = parse_json_args(args, "tools.execute <json>")?;
            request(Method::ToolsExecute, &params)?
        }

        "prefill.markdown.create" => {
            let params: PrefillMarkdownCreateParams =
                parse_json_args(args, "prefill.markdown.create <json>")?;
            request(Method::PrefillMarkdownCreate, &params)?
        }
        "prefill.markdown.delete" => {
            if args.is_empty() {
                println!("Usage: prefill.markdown.delete <filename>");
                return Ok(());
            }
            request(
                Method::PrefillMarkdownDelete,
                &PrefillMarkdownDeleteParams {
                    filename: args.to_string(),
                },
            )?
        }

        "prefill.collection.create" => {
            let params: PrefillCollectionCreateParams =
                parse_json_args(args, "prefill.collection.create <json>")?;
            request(Method::PrefillCollectionCreate, &params)?
        }
        "prefill.collection.delete" => {
            if args.is_empty() {
                println!("Usage: prefill.collection.delete <id>");
                return Ok(());
            }
            request(
                Method::PrefillCollectionDelete,
                &PrefillCollectionDeleteParams {
                    id: args.to_string(),
                },
            )?
        }

        _ => {
            println!("Unknown command: {cmd}. Type 'help' for available commands.");
            return Ok(());
        }
    };

    match serde_json::to_string_pretty(&frame) {
        Ok(json) => println!(">>> {json}"),
        Err(e) => tracing::error!("Failed to format frame: {e}"),
    }

    conn.send_frame(&frame)
        .await
        .map_err(|e| format!("Send failed: {e}"))
}

fn request(
    method: Method,
    params: &impl serde::Serialize,
) -> std::result::Result<Frame, String> {
    Frame::request(method, params).map_err(|e| format!("Failed to build request: {e}"))
}

fn parse_json_args<T: serde::de::DeserializeOwned>(
    args: &str,
    usage: &str,
) -> std::result::Result<T, String> {
    if args.is_empty() {
        println!("Usage: {usage}");
        return Err("Missing arguments".to_string());
    }
    serde_json::from_str(args).map_err(|e| {
        println!("Invalid JSON: {e}");
        format!("JSON parse error: {e}")
    })
}

/// Parse agent arguments supporting `--session <id>` and `--model <id>` flags.
fn parse_agent_args(input: &str) -> (Option<String>, Option<String>, String) {
    let mut session_id = None;
    let mut model_id = None;
    let mut prompt_parts = Vec::new();
    let mut iter = input.split_whitespace().peekable();

    while let Some(token) = iter.next() {
        match token {
            "--session" => {
                session_id = iter.next().map(String::from);
            }
            "--model" => {
                model_id = iter.next().map(String::from);
            }
            _ => prompt_parts.push(token),
        }
    }

    (session_id, model_id, prompt_parts.join(" "))
}

fn uuid_v7() -> String {
    Frame::new_id()
}

fn print_help() {
    println!("Available commands:");
    println!("  health                          - Health check");
    println!("  status                          - Gateway status");
    println!("  agent <prompt>                  - Send agent request");
    println!("    --session <id>                  Resume session");
    println!("    --model <id>                    Use specific model");
    println!("  session.create [name]           - Create session");
    println!("  session.list                    - List sessions");
    println!("  session.get <id>                - Get session messages");
    println!("  session.clear <id>              - Clear session");
    println!("  tools.catalog                   - List available tools");
    println!("  tools.execute <json>            - Execute a tool");
    println!("  cron.create <json>              - Create cron job");
    println!("  cron.list                       - List cron jobs");
    println!("  cron.delete <id>                - Delete cron job");
    println!("  prefill.markdown.create <json>  - Create markdown file");
    println!("  prefill.markdown.list           - List markdown files");
    println!("  prefill.markdown.delete <id>    - Delete markdown file");
    println!("  prefill.collection.create <json> - Create collection");
    println!("  prefill.collection.list         - List collections");
    println!("  prefill.collection.delete <id>  - Delete collection");
    println!("  system-presence [status]        - Report presence");
    println!("  help                            - Show this help");
    println!("  quit                            - Disconnect");
}
