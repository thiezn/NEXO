use std::path::{Path, PathBuf};

use base64::Engine;
use nexo_ws_schema::{
    AgentParams, CronCreateParams, CronDeleteParams, ImageAnalyzeParams,
    PrefillCollectionCreateParams, PrefillCollectionDeleteParams, PrefillMarkdownCreateParams,
    PrefillMarkdownDeleteParams, SendParams, SessionClearParams, SessionCreateParams,
    SessionGetParams, SystemPresenceParams, ToolsExecuteParams,
};

pub struct CommandContext<'a> {
    pub current_session_id: Option<&'a str>,
    pub default_session_name: Option<&'a str>,
    pub default_model_id: Option<&'a str>,
    pub workspace_root: &'a Path,
}

pub const COMMAND_NAMES: &[&str] = &[
    "help",
    "quit",
    "clear",
    "health",
    "status",
    "send",
    "agent",
    "session create",
    "session list",
    "session get",
    "session clear",
    "tools catalog",
    "tools execute",
    "cron create",
    "cron list",
    "cron delete",
    "prefill markdown create",
    "prefill markdown list",
    "prefill markdown delete",
    "prefill collection create",
    "prefill collection list",
    "prefill collection delete",
    "system presence",
    "image analyze",
];

#[derive(Debug, Clone)]
pub enum AppCommand {
    Help,
    Quit,
    Clear,
    Health,
    Status,
    Send(SendParams),
    Agent(AgentParams),
    SessionCreate(SessionCreateParams),
    SessionList,
    SessionGet(SessionGetParams),
    SessionClear(SessionClearParams),
    ToolsCatalog,
    ToolsExecute(ToolsExecuteParams),
    CronCreate(CronCreateParams),
    CronList,
    CronDelete(CronDeleteParams),
    PrefillMarkdownCreate(PrefillMarkdownCreateParams),
    PrefillMarkdownList,
    PrefillMarkdownDelete(PrefillMarkdownDeleteParams),
    PrefillCollectionCreate(PrefillCollectionCreateParams),
    PrefillCollectionList,
    PrefillCollectionDelete(PrefillCollectionDeleteParams),
    SystemPresence(SystemPresenceParams),
    ImageAnalyze(ImageAnalyzeParams),
}

pub fn parse(input: &str, context: CommandContext<'_>) -> Result<AppCommand, String> {
    let trimmed = input.trim();
    let Some(without_slash) = trimmed.strip_prefix('/') else {
        return parse_agent(trimmed, context);
    };

    let Some((command, args)) = split_command(without_slash) else {
        let attempted = without_slash.split_whitespace().next().unwrap_or_default();
        return if attempted.is_empty() {
            Err("Unknown command. Use /help to see available commands.".to_string())
        } else {
            Err(format!(
                "Unknown command '/{attempted}'. Use /help to see available commands."
            ))
        };
    };

    match command {
        "help" => Ok(AppCommand::Help),
        "quit" => Ok(AppCommand::Quit),
        "clear" => Ok(AppCommand::Clear),
        "health" => Ok(AppCommand::Health),
        "status" => Ok(AppCommand::Status),
        "send" => parse_send(args),
        "agent" => parse_agent(args, context),
        "session create" => Ok(AppCommand::SessionCreate(SessionCreateParams {
            name: if args.is_empty() {
                context.default_session_name.map(ToOwned::to_owned)
            } else {
                Some(args.to_string())
            },
            prefill_collection_id: None,
        })),
        "session list" => Ok(AppCommand::SessionList),
        "session get" => required_arg(args, "/session get <session-id>")
            .map(|session_id| AppCommand::SessionGet(SessionGetParams { session_id })),
        "session clear" => required_arg(args, "/session clear <session-id>")
            .map(|session_id| AppCommand::SessionClear(SessionClearParams { session_id })),
        "tools catalog" => Ok(AppCommand::ToolsCatalog),
        "tools execute" => {
            parse_json_args(args, "/tools execute <json>").map(AppCommand::ToolsExecute)
        }
        "cron create" => parse_json_args(args, "/cron create <json>").map(AppCommand::CronCreate),
        "cron list" => Ok(AppCommand::CronList),
        "cron delete" => required_arg(args, "/cron delete <job-id>")
            .map(|job_id| AppCommand::CronDelete(CronDeleteParams { job_id })),
        "prefill markdown create" => parse_json_args(args, "/prefill markdown create <json>")
            .map(AppCommand::PrefillMarkdownCreate),
        "prefill markdown list" => Ok(AppCommand::PrefillMarkdownList),
        "prefill markdown delete" => {
            required_arg(args, "/prefill markdown delete <filename>").map(|filename| {
                AppCommand::PrefillMarkdownDelete(PrefillMarkdownDeleteParams { filename })
            })
        }
        "prefill collection create" => parse_json_args(args, "/prefill collection create <json>")
            .map(AppCommand::PrefillCollectionCreate),
        "prefill collection list" => Ok(AppCommand::PrefillCollectionList),
        "prefill collection delete" => required_arg(args, "/prefill collection delete <id>")
            .map(|id| AppCommand::PrefillCollectionDelete(PrefillCollectionDeleteParams { id })),
        "system presence" => Ok(AppCommand::SystemPresence(SystemPresenceParams {
            status: if args.is_empty() {
                "active".to_string()
            } else {
                args.to_string()
            },
        })),
        "image analyze" => parse_image_analyze(args, context.workspace_root),
        _ => Err(format!(
            "Unknown command '/{command}'. Use /help to see available commands."
        )),
    }
}

pub fn help_text() -> &'static str {
    "Available commands:

Core:
/help
/quit
/clear
/health
/status

Messaging:
/send <target> <json>
/agent [--session <id>] [--model <id>] <prompt>
/image analyze <@image-path|path> <prompt>

Sessions:
/session create [name]
/session list
/session get <session-id>
/session clear <session-id>

Tools and jobs:
/tools catalog
/tools execute <json>
/cron create <json>
/cron list
/cron delete <job-id>

Prefills:
/prefill markdown create <json>
/prefill markdown list
/prefill markdown delete <filename>
/prefill collection create <json>
/prefill collection list
/prefill collection delete <id>

Presence:
/system presence [status]

Autocomplete:
- Use Tab to accept the current suggestion.
- Use Up/Down or Shift+Tab to cycle suggestions.
- Type plain text to run it as `/agent <prompt>`.
- Use @path in /agent prompts to inline file contents.
- Use @path as the image argument for /image analyze.
- Press F1 or type /help to open this help view.
- Press Esc to close the help view or dismiss autocomplete.
"
}

fn split_command(input: &str) -> Option<(&'static str, &str)> {
    for command in COMMAND_NAMES {
        if let Some(args) = strip_command_prefix(input, command) {
            return Some((*command, args));
        }
    }

    strip_command_prefix(input, "exit").map(|args| ("quit", args))
}

fn strip_command_prefix<'a>(input: &'a str, command: &str) -> Option<&'a str> {
    if input == command {
        return Some("");
    }

    let rest = input.strip_prefix(command)?;
    rest.chars()
        .next()
        .is_some_and(char::is_whitespace)
        .then_some(rest.trim_start())
}

fn parse_send(args: &str) -> Result<AppCommand, String> {
    let Some((target, payload)) = args.split_once(' ') else {
        return Err("Usage: /send <target> <json>".into());
    };

    let payload = serde_json::from_str(payload.trim())
        .map_err(|e| format!("Invalid JSON payload for /send: {e}"))?;
    Ok(AppCommand::Send(SendParams {
        target: target.trim().to_string(),
        payload,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn parse_agent(args: &str, context: CommandContext<'_>) -> Result<AppCommand, String> {
    if args.is_empty() {
        return Err("Usage: /agent [--session <id>] [--model <id>] <prompt>".into());
    }

    let mut session_id = context.current_session_id.map(ToOwned::to_owned);
    let mut model_id = context.default_model_id.map(ToOwned::to_owned);
    let mut prompt_parts = Vec::new();
    let mut iter = args.split_whitespace().peekable();

    while let Some(token) = iter.next() {
        match token {
            "--session" => {
                session_id = Some(
                    iter.next()
                        .map(ToOwned::to_owned)
                        .ok_or_else(|| "Missing value for --session".to_string())?,
                );
            }
            "--model" => {
                model_id = Some(
                    iter.next()
                        .map(ToOwned::to_owned)
                        .ok_or_else(|| "Missing value for --model".to_string())?,
                );
            }
            _ => prompt_parts.push(token),
        }
    }

    if prompt_parts.is_empty() {
        return Err("Usage: /agent [--session <id>] [--model <id>] <prompt>".into());
    }

    let prompt = expand_prompt_file_references(&prompt_parts.join(" "), context.workspace_root)?;
    Ok(AppCommand::Agent(AgentParams {
        prompt,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
        session_id,
        context: None,
        model_id,
        thinking: None,
    }))
}

fn parse_image_analyze(args: &str, workspace_root: &Path) -> Result<AppCommand, String> {
    let Some((image_path, prompt)) = args.split_once(' ') else {
        return Err("Usage: /image analyze <@image-path|path> <prompt>".into());
    };

    let image_path = resolve_path(workspace_root, image_path)?;
    let image_bytes = std::fs::read(&image_path)
        .map_err(|e| format!("Failed to read image '{}': {e}", image_path.display()))?;
    let image_data = base64::engine::general_purpose::STANDARD.encode(image_bytes);

    Ok(AppCommand::ImageAnalyze(ImageAnalyzeParams {
        image_data,
        prompt: prompt.trim().to_string(),
        max_tokens: 4096,
        temperature: 1.0,
        visual_token_budget: None,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn parse_json_args<T: serde::de::DeserializeOwned>(args: &str, usage: &str) -> Result<T, String> {
    if args.is_empty() {
        return Err(format!("Usage: {usage}"));
    }
    serde_json::from_str(args).map_err(|e| format!("Invalid JSON for {usage}: {e}"))
}

fn required_arg(args: &str, usage: &str) -> Result<String, String> {
    if args.is_empty() {
        return Err(format!("Usage: {usage}"));
    }
    Ok(args.to_string())
}

fn expand_prompt_file_references(input: &str, workspace_root: &Path) -> Result<String, String> {
    let mut output = Vec::new();
    for token in input.split_whitespace() {
        if let Some(path_token) = token.strip_prefix('@') {
            let path = resolve_path(workspace_root, path_token)?;
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read '{}': {e}", path.display()))?;
            let display_path = display_path(workspace_root, &path);
            let content = truncate_large_text(&content);
            output.push(format!(
                "\n[File: {display_path}]\n```text\n{content}\n```\n"
            ));
        } else {
            output.push(token.to_string());
        }
    }
    Ok(output.join(" "))
}

fn truncate_large_text(input: &str) -> String {
    const MAX_CHARS: usize = 24_000;
    if input.chars().count() <= MAX_CHARS {
        return input.to_string();
    }

    let truncated: String = input.chars().take(MAX_CHARS).collect();
    format!("{truncated}\n...[truncated]...")
}

fn resolve_path(workspace_root: &Path, token: &str) -> Result<PathBuf, String> {
    let token = token.trim_start_matches('@');
    let path = Path::new(token);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    };

    resolved
        .canonicalize()
        .map_err(|e| format!("Failed to resolve '{}': {e}", resolved.display()))
}

fn display_path(workspace_root: &Path, path: &Path) -> String {
    let canonical_root = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());

    path.strip_prefix(&canonical_root)
        .or_else(|_| path.strip_prefix(workspace_root))
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::fs;

    use super::*;

    fn context<'a>(root: &'a Path) -> CommandContext<'a> {
        CommandContext {
            current_session_id: Some("sess-1"),
            default_session_name: Some("cli-session"),
            default_model_id: Some("gemma-4"),
            workspace_root: root,
        }
    }

    #[test]
    fn parses_send_command() {
        let command = parse("/send bob {\"text\":\"hello\"}", context(Path::new("."))).unwrap();
        match command {
            AppCommand::Send(params) => {
                assert_eq!(params.target, "bob");
                assert_eq!(params.payload["text"], "hello");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_agent_with_default_context() {
        let command = parse("/agent summarize this", context(Path::new("."))).unwrap();
        match command {
            AppCommand::Agent(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.model_id.as_deref(), Some("gemma-4"));
                assert_eq!(params.prompt, "summarize this");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_plain_text_as_agent_command() {
        let command = parse("summarize this", context(Path::new("."))).unwrap();
        match command {
            AppCommand::Agent(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.model_id.as_deref(), Some("gemma-4"));
                assert_eq!(params.prompt, "summarize this");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn expands_agent_file_references() {
        let temp_root =
            std::env::temp_dir().join(format!("nexo-client-command-test-{}", std::process::id()));
        fs::create_dir_all(&temp_root).unwrap();
        let file_path = temp_root.join("sample.txt");
        fs::write(&file_path, "hello world").unwrap();

        let command = parse("/agent inspect @sample.txt", context(&temp_root)).unwrap();
        match command {
            AppCommand::Agent(params) => {
                assert!(params.prompt.contains("File: sample.txt"));
                assert!(params.prompt.contains("hello world"));
            }
            other => panic!("unexpected command: {other:?}"),
        }

        let _ = fs::remove_file(file_path);
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn uses_default_session_name_for_session_create() {
        let command = parse("/session create", context(Path::new("."))).unwrap();
        match command {
            AppCommand::SessionCreate(params) => {
                assert_eq!(params.name.as_deref(), Some("cli-session"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_spaced_session_list_command() {
        let command = parse("/session list", context(Path::new("."))).unwrap();
        assert!(matches!(command, AppCommand::SessionList));
    }
}
