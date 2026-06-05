use std::path::{Path, PathBuf};

use nexo_ws_schema::{
    AudioAnalyzeParams, AudioGenerateParams, CronCreateParams, CronDeleteParams,
    ImageAnalyzeParams, ImageGenerateParams, PromptCollectionCreateParams,
    PromptCollectionDeleteParams, PromptDocumentCreateParams, PromptDocumentDeleteParams,
    RunStartParams, SendParams, SessionClearParams, SessionCreateParams, SessionGetParams,
    SystemPresenceParams, ToolsExecuteParams,
};

use crate::audio;

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
    "run",
    "session create",
    "session list",
    "session get",
    "session clear",
    "tools catalog",
    "tools execute",
    "cron create",
    "cron list",
    "cron delete",
    "prompt document create",
    "prompt document list",
    "prompt document delete",
    "prompt collection create",
    "prompt collection list",
    "prompt collection delete",
    "system presence",
    "analyze image",
    "analyze audio",
    "generate image",
    "generate audio",
];

#[derive(Debug, Clone)]
pub enum AppCommand {
    Help,
    Quit,
    Clear,
    Health,
    Status,
    Send(SendParams),
    RunStart(RunStartParams),
    SessionCreate(SessionCreateParams),
    SessionList,
    SessionGet(SessionGetParams),
    SessionClear(SessionClearParams),
    ToolsCatalog,
    ToolsExecute(ToolsExecuteParams),
    CronCreate(CronCreateParams),
    CronList,
    CronDelete(CronDeleteParams),
    PromptDocumentCreate(PromptDocumentCreateParams),
    PromptDocumentList,
    PromptDocumentDelete(PromptDocumentDeleteParams),
    PromptCollectionCreate(PromptCollectionCreateParams),
    PromptCollectionList,
    PromptCollectionDelete(PromptCollectionDeleteParams),
    SystemPresence(SystemPresenceParams),
    ImageAnalyze(ImageAnalyzeParams),
    AudioAnalyze(AudioAnalyzeParams),
    ImageGenerate(ImageGenerateParams),
    AudioGenerate(AudioGenerateParams),
}

pub fn parse(input: &str, context: CommandContext<'_>) -> Result<AppCommand, String> {
    let trimmed = input.trim();
    let Some(without_slash) = trimmed.strip_prefix('/') else {
        return parse_run(trimmed, context);
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
        "run" => parse_run(args, context),
        "session create" => Ok(AppCommand::SessionCreate(SessionCreateParams {
            name: if args.is_empty() {
                context.default_session_name.map(ToOwned::to_owned)
            } else {
                Some(args.to_string())
            },
            prompt_collection_id: None,
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
        "prompt document create" => parse_json_args(args, "/prompt document create <json>")
            .map(AppCommand::PromptDocumentCreate),
        "prompt document list" => Ok(AppCommand::PromptDocumentList),
        "prompt document delete" => required_arg(args, "/prompt document delete <id>")
            .map(|id| AppCommand::PromptDocumentDelete(PromptDocumentDeleteParams { id })),
        "prompt collection create" => parse_json_args(args, "/prompt collection create <json>")
            .map(AppCommand::PromptCollectionCreate),
        "prompt collection list" => Ok(AppCommand::PromptCollectionList),
        "prompt collection delete" => required_arg(args, "/prompt collection delete <id>")
            .map(|id| AppCommand::PromptCollectionDelete(PromptCollectionDeleteParams { id })),
        "system presence" => Ok(AppCommand::SystemPresence(SystemPresenceParams {
            status: if args.is_empty() {
                "active".to_string()
            } else {
                args.to_string()
            },
        })),
        "analyze image" => {
            parse_image_analyze(args, context.workspace_root, context.current_session_id)
        }
        "analyze audio" => {
            parse_audio_analyze(args, context.workspace_root, context.current_session_id)
        }
        "generate image" => parse_image_generate(args, context.current_session_id),
        "generate audio" => parse_audio_generate(args, context.current_session_id),
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
/run [--session <id>] [--model <id>] <prompt>
/analyze image <@image-path|path> <prompt>
/analyze audio <@audio-path|path> <prompt>
/analyze audio --mic [--max-secs <seconds>] <prompt>
/generate image <prompt>
/generate audio <prompt>

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

Prompts:
/prompt document create <json>
/prompt document list
/prompt document delete <id>
/prompt collection create <json>
/prompt collection list
/prompt collection delete <id>

Presence:
/system presence [status]

Autocomplete:
- Use Tab to accept the current suggestion.
- Use Up/Down or Shift+Tab to cycle suggestions.
- Type plain text to run it as `/run <prompt>`.
- Use @path in /run prompts to inline file contents.
- Use @path as the image argument for /analyze image.
- Use @path as the audio argument for /analyze audio.
- Use /analyze audio --mic to record from microphone, then analyze.
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

fn parse_run(args: &str, context: CommandContext<'_>) -> Result<AppCommand, String> {
    if args.is_empty() {
        return Err("Usage: /run [--session <id>] [--model <id>] <prompt>".into());
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
        return Err("Usage: /run [--session <id>] [--model <id>] <prompt>".into());
    }

    let prompt = expand_prompt_file_references(&prompt_parts.join(" "), context.workspace_root)?;
    Ok(AppCommand::RunStart(RunStartParams {
        input: prompt,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
        session_id,
        instructions: None,
        model_id,
        reasoning: None,
        tool_choice: None,
    }))
}

fn parse_image_analyze(
    args: &str,
    workspace_root: &Path,
    current_session_id: Option<&str>,
) -> Result<AppCommand, String> {
    let Some((image_path, prompt)) = args.split_once(' ') else {
        return Err("Usage: /analyze image <@image-path|path> <prompt>".into());
    };

    let image_path = resolve_path(workspace_root, image_path)?;
    let image_bytes = std::fs::read(&image_path)
        .map_err(|e| format!("Failed to read image '{}': {e}", image_path.display()))?;
    let image_data = encode_base64(&image_bytes);

    Ok(AppCommand::ImageAnalyze(ImageAnalyzeParams {
        image_data,
        session_id: current_session_id.map(ToOwned::to_owned),
        media_type: detect_image_media_type(&image_path),
        prompt: prompt.trim().to_string(),
        max_tokens: 4096,
        temperature: 1.0,
        visual_token_budget: None,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn parse_audio_analyze(
    args: &str,
    workspace_root: &Path,
    current_session_id: Option<&str>,
) -> Result<AppCommand, String> {
    if args.is_empty() {
        return Err(
            "Usage: /analyze audio <@audio-path|path> <prompt> OR /analyze audio --mic [--max-secs <seconds>] <prompt>"
                .into(),
        );
    }

    let trimmed = args.trim();
    if let Some(rest) = trimmed.strip_prefix("--mic") {
        return parse_audio_analyze_mic(rest.trim_start(), current_session_id);
    }

    let Some((audio_path, prompt)) = trimmed.split_once(' ') else {
        return Err("Usage: /analyze audio <@audio-path|path> <prompt>".into());
    };
    let audio_path = resolve_path(workspace_root, audio_path)?;
    let audio_bytes = std::fs::read(&audio_path)
        .map_err(|e| format!("Failed to read audio '{}': {e}", audio_path.display()))?;
    let buffer = audio::load_file(&audio_path)
        .map_err(|e| format!("Failed to decode audio '{}': {e}", audio_path.display()))?;

    Ok(AppCommand::AudioAnalyze(AudioAnalyzeParams {
        audio_data: encode_base64(&audio_bytes),
        session_id: current_session_id.map(ToOwned::to_owned),
        media_type: detect_audio_media_type(&audio_path),
        sample_rate_hz: Some(buffer.sample_rate),
        channel_count: Some(buffer.channels),
        prompt: prompt.trim().to_string(),
        max_tokens: 4096,
        temperature: 1.0,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn parse_audio_analyze_mic(
    args: &str,
    current_session_id: Option<&str>,
) -> Result<AppCommand, String> {
    let mut max_secs = 8.0;
    let mut prompt_parts = Vec::new();
    let mut iter = args.split_whitespace().peekable();

    while let Some(token) = iter.next() {
        match token {
            "--max-secs" => {
                let value = iter.next().ok_or_else(|| {
                    "Missing value for --max-secs in /analyze audio --mic".to_string()
                })?;
                max_secs = value
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid --max-secs value '{value}'"))?;
                if max_secs <= 0.0 {
                    return Err("--max-secs must be greater than 0".to_string());
                }
            }
            _ => prompt_parts.push(token),
        }
    }

    if prompt_parts.is_empty() {
        return Err("Usage: /analyze audio --mic [--max-secs <seconds>] <prompt>".into());
    }

    let config = audio::RecordConfig {
        sample_rate: 16_000,
        max_duration_secs: Some(max_secs),
        silence_threshold_secs: Some(2.0),
        silence_rms_threshold: 0.01,
    };
    let buffer = audio::record_microphone(&config)
        .map_err(|e| format!("Failed to record microphone audio: {e}"))?
        .to_mono();
    let wav_bytes =
        audio::encode_wav(&buffer).map_err(|e| format!("Failed to encode recorded audio: {e}"))?;

    Ok(AppCommand::AudioAnalyze(AudioAnalyzeParams {
        audio_data: encode_base64(&wav_bytes),
        session_id: current_session_id.map(ToOwned::to_owned),
        media_type: Some("audio/wav".to_string()),
        sample_rate_hz: Some(buffer.sample_rate),
        channel_count: Some(buffer.channels),
        prompt: prompt_parts.join(" "),
        max_tokens: 4096,
        temperature: 1.0,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn encode_base64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn detect_image_media_type(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let media_type = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "avif" => "image/avif",
        "heic" => "image/heic",
        _ => return None,
    };
    Some(media_type.to_string())
}

fn detect_audio_media_type(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let media_type = match extension.as_str() {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "opus" => "audio/opus",
        "m4a" => "audio/mp4",
        "aac" => "audio/aac",
        "webm" => "audio/webm",
        _ => return None,
    };
    Some(media_type.to_string())
}

fn parse_image_generate(
    args: &str,
    current_session_id: Option<&str>,
) -> Result<AppCommand, String> {
    let prompt = args.trim();
    if prompt.is_empty() {
        return Err("Usage: /generate image <prompt>".to_string());
    }

    Ok(AppCommand::ImageGenerate(ImageGenerateParams {
        prompt: prompt.to_string(),
        session_id: current_session_id.map(ToOwned::to_owned),
        negative_prompt: None,
        width: 1024,
        height: 1024,
        sample_count: 1,
        steps: None,
        guidance_scale: None,
        seed: None,
        idempotency_key: nexo_ws_schema::Frame::new_id(),
    }))
}

fn parse_audio_generate(
    args: &str,
    current_session_id: Option<&str>,
) -> Result<AppCommand, String> {
    let prompt = args.trim();
    if prompt.is_empty() {
        return Err("Usage: /generate audio <prompt>".to_string());
    }

    Ok(AppCommand::AudioGenerate(AudioGenerateParams {
        prompt: prompt.to_string(),
        session_id: current_session_id.map(ToOwned::to_owned),
        language: Default::default(),
        voice: None,
        sample_rate_hz: None,
        speed: None,
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
    fn parses_run_with_default_context() {
        let command = parse("/run summarize this", context(Path::new("."))).unwrap();
        match command {
            AppCommand::RunStart(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.model_id.as_deref(), Some("gemma-4"));
                assert_eq!(params.input, "summarize this");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_plain_text_as_run_command() {
        let command = parse("summarize this", context(Path::new("."))).unwrap();
        match command {
            AppCommand::RunStart(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.model_id.as_deref(), Some("gemma-4"));
                assert_eq!(params.input, "summarize this");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn expands_run_file_references() {
        let temp_root =
            std::env::temp_dir().join(format!("nexo-client-command-test-{}", std::process::id()));
        fs::create_dir_all(&temp_root).unwrap();
        let file_path = temp_root.join("sample.txt");
        fs::write(&file_path, "hello world").unwrap();

        let command = parse("/run inspect @sample.txt", context(&temp_root)).unwrap();
        match command {
            AppCommand::RunStart(params) => {
                assert!(params.input.contains("File: sample.txt"));
                assert!(params.input.contains("hello world"));
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

    #[test]
    fn parses_image_analyze_with_media_type() {
        let temp_root =
            std::env::temp_dir().join(format!("nexo-client-image-test-{}", std::process::id()));
        fs::create_dir_all(&temp_root).unwrap();
        let image_path = temp_root.join("sample.png");

        let image = image::RgbImage::from_vec(1, 1, vec![255, 0, 0]).unwrap();
        image.save(&image_path).unwrap();

        let command = parse(
            "/analyze image @sample.png what is this?",
            context(&temp_root),
        )
        .unwrap();

        match command {
            AppCommand::ImageAnalyze(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.media_type.as_deref(), Some("image/png"));
                assert_eq!(params.prompt, "what is this?");
                assert!(!params.image_data.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }

        let _ = fs::remove_file(image_path);
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn parses_audio_analyze_from_file() {
        let temp_root =
            std::env::temp_dir().join(format!("nexo-client-audio-test-{}", std::process::id()));
        fs::create_dir_all(&temp_root).unwrap();
        let audio_path = temp_root.join("sample.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&audio_path, spec).unwrap();
            for _ in 0..1600 {
                writer.write_sample(0_i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let command = parse(
            "/analyze audio @sample.wav summarize this clip",
            context(&temp_root),
        )
        .unwrap();

        match command {
            AppCommand::AudioAnalyze(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.media_type.as_deref(), Some("audio/wav"));
                assert_eq!(params.sample_rate_hz, Some(16_000));
                assert_eq!(params.channel_count, Some(1));
                assert_eq!(params.prompt, "summarize this clip");
                assert!(!params.audio_data.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }

        let _ = fs::remove_file(audio_path);
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn parses_generate_image_prompt() {
        let command = parse(
            "/generate image paint a red lighthouse",
            context(Path::new(".")),
        )
        .unwrap();

        match command {
            AppCommand::ImageGenerate(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.prompt, "paint a red lighthouse");
                assert_eq!(params.width, 1024);
                assert_eq!(params.height, 1024);
                assert_eq!(params.sample_count, 1);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_generate_audio_prompt() {
        let command = parse(
            "/generate audio ocean waves at sunset",
            context(Path::new(".")),
        )
        .unwrap();

        match command {
            AppCommand::AudioGenerate(params) => {
                assert_eq!(params.session_id.as_deref(), Some("sess-1"));
                assert_eq!(params.prompt, "ocean waves at sunset");
                assert!(params.voice.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn detects_media_types_from_file_extensions() {
        assert_eq!(
            detect_image_media_type(Path::new("photo.jpeg")).as_deref(),
            Some("image/jpeg")
        );
        assert_eq!(
            detect_audio_media_type(Path::new("speech.m4a")).as_deref(),
            Some("audio/mp4")
        );
        assert_eq!(detect_image_media_type(Path::new("data.unknown")), None);
        assert_eq!(detect_audio_media_type(Path::new("data.unknown")), None);
    }
}
