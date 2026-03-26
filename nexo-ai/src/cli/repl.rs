use crate::coordinator::Coordinator;
use crate::registry::{find_manifest, known_manifests};
use crate::shared::templates::conversation::ConversationManager;
use crate::shared::types::{ChatMessage, ChatRole, ModelCategory};
use crate::statistics::display as stats_display;
use anyhow::Result;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper};

// ── Commands ────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
pub enum ReplCommand {
    Chat { text: String },
    Tool { text: String },
    Talk { text: String },
    Listen { file: Option<String> },
    Imagine { prompt: String },
    Image { path: String, prompt: String },
    Embed { text: String },
    LoadCategories { categories: Vec<String> },
    LoadModels { models: Vec<String> },
    UnloadModels { models: Vec<String> },
    UnloadCategories { categories: Vec<String> },
    UnloadAll,
    Set { key: String, value: String },
    Get { key: Option<String> },
    ListModels { model: Option<String> },
    Stats { model: Option<String> },
    Clear,
    Help { command: Option<String> },
    Quit,
    Empty,
    Unknown(String),
    Ping,
}

// ── Parser ──────────────────────────────────────────────────────────────────

fn parse_comma_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn parse_repl_input(input: &str) -> ReplCommand {
    let input = input.trim();
    if input.is_empty() {
        return ReplCommand::Empty;
    }

    if !input.starts_with('/') {
        return ReplCommand::Chat {
            text: input.to_string(),
        };
    }

    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        "/chat" => ReplCommand::Chat {
            text: args.to_string(),
        },
        "/tool" => ReplCommand::Tool {
            text: args.to_string(),
        },
        "/talk" => ReplCommand::Talk {
            text: args.to_string(),
        },
        "/listen" => {
            if args.is_empty() {
                ReplCommand::Listen { file: None }
            } else {
                ReplCommand::Listen {
                    file: Some(args.to_string()),
                }
            }
        }
        "/imagine" => ReplCommand::Imagine {
            prompt: args.to_string(),
        },
        "/embed" => ReplCommand::Embed {
            text: args.to_string(),
        },
        "/image" => {
            let img_parts: Vec<&str> = args.splitn(2, ' ').collect();
            if img_parts.len() == 2 {
                ReplCommand::Image {
                    path: img_parts[0].to_string(),
                    prompt: img_parts[1].to_string(),
                }
            } else {
                ReplCommand::Unknown(input.to_string())
            }
        }
        "/load" => {
            if let Some(rest) = args.strip_prefix("models ") {
                ReplCommand::LoadModels {
                    models: parse_comma_list(rest),
                }
            } else if let Some(rest) = args.strip_prefix("categories ") {
                ReplCommand::LoadCategories {
                    categories: parse_comma_list(rest),
                }
            } else if !args.is_empty() {
                // Bare args treated as categories for backwards compat.
                ReplCommand::LoadCategories {
                    categories: parse_comma_list(args),
                }
            } else {
                ReplCommand::Unknown(input.to_string())
            }
        }
        "/unload" => {
            if let Some(rest) = args.strip_prefix("models ") {
                ReplCommand::UnloadModels {
                    models: parse_comma_list(rest),
                }
            } else if let Some(rest) = args.strip_prefix("categories ") {
                ReplCommand::UnloadCategories {
                    categories: parse_comma_list(rest),
                }
            } else if args == "all" {
                ReplCommand::UnloadAll
            } else {
                ReplCommand::Unknown(input.to_string())
            }
        }
        "/set" => {
            if args.is_empty() {
                ReplCommand::Get { key: None }
            } else {
                let set_parts: Vec<&str> = args.splitn(2, ' ').collect();
                if set_parts.len() == 2 {
                    ReplCommand::Set {
                        key: set_parts[0].to_string(),
                        value: set_parts[1].trim_matches('"').to_string(),
                    }
                } else {
                    ReplCommand::Get {
                        key: Some(args.to_string()),
                    }
                }
            }
        }
        "/get" => {
            if args.is_empty() {
                ReplCommand::Get { key: None }
            } else {
                ReplCommand::Get {
                    key: Some(args.to_string()),
                }
            }
        }
        "/stats" => {
            let model = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            ReplCommand::Stats { model }
        }
        "/list" => {
            if args.is_empty() {
                ReplCommand::ListModels { model: None }
            } else {
                ReplCommand::ListModels {
                    model: Some(args.to_string()),
                }
            }
        }
        "/help" | "/h" | "/?" => {
            if args.is_empty() {
                ReplCommand::Help { command: None }
            } else {
                ReplCommand::Help {
                    command: Some(args.to_string()),
                }
            }
        }
        "/clear" => ReplCommand::Clear,
        "/quit" | "/q" | "/exit" => ReplCommand::Quit,
        _ => ReplCommand::Unknown(input.to_string()),
    }
}

// ── Auto-completion ─────────────────────────────────────────────────────────

const COMMANDS: &[&str] = &[
    "/chat", "/tool", "/talk", "/listen", "/imagine", "/image", "/embed", "/load", "/unload",
    "/set", "/get", "/list", "/stats", "/clear", "/help", "/quit", "/exit", "/ping",
];

const HELP_TOPICS: &[&str] = &[
    "chat", "tool", "talk", "listen", "imagine", "image", "embed", "load", "unload", "set", "get",
    "list", "stats", "clear", "quit",
];

struct CompletionData {
    model_names: Vec<String>,
    category_names: Vec<String>,
    config_keys: Vec<String>,
}

impl CompletionData {
    fn from_coordinator(coordinator: &Coordinator) -> Self {
        let model_names: Vec<String> = known_manifests()
            .iter()
            .map(|m| m.manifest.name.clone())
            .collect();
        let category_names: Vec<String> = ModelCategory::all()
            .iter()
            .map(|c| c.as_str().to_string())
            .collect();
        let mut config_keys: Vec<String> = category_names
            .iter()
            .map(|c| format!("default-{c}"))
            .collect();
        config_keys.push("startup-categories".to_string());
        // Add model-specific setting keys for known models.
        for name in &model_names {
            if coordinator.config().models.contains_key(name) {
                for suffix in &[
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "seed",
                    "voice_description",
                ] {
                    config_keys.push(format!("{name}.{suffix}"));
                }
            }
        }
        Self {
            model_names,
            category_names,
            config_keys,
        }
    }
}

struct ReplHelper {
    data: CompletionData,
    file_completer: rustyline::completion::FilenameCompleter,
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let trimmed = line[..pos].trim_start();
        // Delegate to file completer for /listen and /image commands.
        if trimmed.starts_with("/listen ") || trimmed.starts_with("/image ") {
            // For /image, only complete the first argument (file path).
            if trimmed.starts_with("/image ") {
                let after_cmd = &trimmed["/image ".len()..];
                // If there's already a space after the first arg, we're on the prompt — no completion.
                if after_cmd.contains(' ') {
                    return Ok((pos, vec![]));
                }
            }
            return self.file_completer.complete(line, pos, ctx);
        }
        Ok(complete_repl(&line[..pos], &self.data))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;
}

impl Highlighter for ReplHelper {}
impl Validator for ReplHelper {}
impl Helper for ReplHelper {}

fn complete_repl(line: &str, data: &CompletionData) -> (usize, Vec<Pair>) {
    if !line.starts_with('/') {
        return (0, vec![]);
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    let trailing_space = line.ends_with(' ');

    match parts.len() {
        0 => (0, vec![]),
        1 if !trailing_space => {
            // Completing the command name.
            let prefix = parts[0];
            let candidates: Vec<Pair> = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: format!("{cmd} "),
                })
                .collect();
            (0, candidates)
        }
        _ => {
            let cmd = parts[0];
            complete_args(cmd, &parts[1..], trailing_space, line, data)
        }
    }
}

fn complete_args(
    cmd: &str,
    args: &[&str],
    trailing_space: bool,
    line: &str,
    data: &CompletionData,
) -> (usize, Vec<Pair>) {
    match cmd {
        "/list" | "/stats" => complete_from_list(args, trailing_space, line, &data.model_names),
        "/load" => {
            if args.is_empty() || (args.len() == 1 && !trailing_space) {
                let subs: Vec<String> = ["categories", "models"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                complete_from_list(args, trailing_space, line, &subs)
            } else {
                match args[0] {
                    "categories" => complete_comma_sep(
                        args.get(1).copied().unwrap_or(""),
                        trailing_space,
                        line,
                        &data.category_names,
                    ),
                    "models" => complete_comma_sep(
                        args.get(1).copied().unwrap_or(""),
                        trailing_space,
                        line,
                        &data.model_names,
                    ),
                    _ => (line.len(), vec![]),
                }
            }
        }
        "/unload" => {
            if args.is_empty() || (args.len() == 1 && !trailing_space) {
                let subs: Vec<String> = ["models", "categories", "all"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                complete_from_list(args, trailing_space, line, &subs)
            } else {
                match args[0] {
                    "categories" => complete_comma_sep(
                        args.get(1).copied().unwrap_or(""),
                        trailing_space,
                        line,
                        &data.category_names,
                    ),
                    "models" => complete_comma_sep(
                        args.get(1).copied().unwrap_or(""),
                        trailing_space,
                        line,
                        &data.model_names,
                    ),
                    _ => (line.len(), vec![]),
                }
            }
        }
        "/set" | "/get" => complete_from_list(args, trailing_space, line, &data.config_keys),
        "/help" => {
            let topics: Vec<String> = HELP_TOPICS.iter().map(|s| s.to_string()).collect();
            complete_from_list(args, trailing_space, line, &topics)
        }
        _ => (line.len(), vec![]),
    }
}

fn complete_from_list(
    args: &[&str],
    trailing_space: bool,
    line: &str,
    options: &[String],
) -> (usize, Vec<Pair>) {
    let partial = if trailing_space || args.is_empty() {
        None
    } else {
        args.last().copied()
    };

    match partial {
        None => {
            let start = line.len();
            let candidates = options
                .iter()
                .map(|o| Pair {
                    display: o.clone(),
                    replacement: o.clone(),
                })
                .collect();
            (start, candidates)
        }
        Some(partial) => {
            let start = line.len() - partial.len();
            let candidates = options
                .iter()
                .filter(|o| o.starts_with(partial))
                .map(|o| Pair {
                    display: o.clone(),
                    replacement: o.clone(),
                })
                .collect();
            (start, candidates)
        }
    }
}

fn complete_comma_sep(
    arg: &str,
    trailing_space: bool,
    line: &str,
    options: &[String],
) -> (usize, Vec<Pair>) {
    let partial = if trailing_space || arg.is_empty() {
        ""
    } else {
        arg.rsplit(',').next().unwrap_or(arg)
    };
    let start = line.len() - partial.len();
    let candidates = options
        .iter()
        .filter(|o| o.starts_with(partial))
        .map(|o| Pair {
            display: o.clone(),
            replacement: o.clone(),
        })
        .collect();
    (start, candidates)
}

// ── REPL main loop ──────────────────────────────────────────────────────────

pub fn run_repl(
    coordinator: &mut Coordinator,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();

    let helper = ReplHelper {
        data: CompletionData::from_coordinator(coordinator),
        file_completer: rustyline::completion::FilenameCompleter::new(),
    };

    let mut rl: rustyline::Editor<ReplHelper, rustyline::history::DefaultHistory> =
        rustyline::Editor::with_config(config)
            .map_err(|e| anyhow::anyhow!("failed to initialize REPL: {e}"))?;
    rl.set_helper(Some(helper));

    let mut conversation = ConversationManager::new();

    println!("nexo-ai REPL. Type /help for commands, /quit to exit.");
    println!();

    let refresh = |rl: &mut rustyline::Editor<ReplHelper, rustyline::history::DefaultHistory>,
                    coord: &Coordinator| {
        if let Some(helper) = rl.helper_mut() {
            helper.data = CompletionData::from_coordinator(coord);
        }
    };

    loop {
        let readline = rl.readline("nexo> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                let command = parse_repl_input(&line);
                let needs_refresh = matches!(
                    command,
                    ReplCommand::LoadCategories { .. }
                        | ReplCommand::LoadModels { .. }
                        | ReplCommand::UnloadModels { .. }
                        | ReplCommand::UnloadCategories { .. }
                        | ReplCommand::UnloadAll
                        | ReplCommand::Set { .. }
                );
                match command {
                    ReplCommand::Quit => {
                        coordinator.unload_all();
                        println!("goodbye.");
                        break;
                    }
                    ReplCommand::Help { command } => print_help(command.as_deref()),
                    ReplCommand::Empty => {}
                    ReplCommand::Stats { model } => {
                        handle_stats(coordinator, model.as_deref());
                    }
                    ReplCommand::ListModels { model } => {
                        handle_list_models(coordinator, model.as_deref());
                    }
                    ReplCommand::LoadCategories { categories } => {
                        handle_load_categories(coordinator, &categories);
                    }
                    ReplCommand::LoadModels { models } => {
                        handle_load_models(coordinator, &models);
                    }
                    ReplCommand::UnloadModels { models } => {
                        handle_unload_models(coordinator, &models);
                    }
                    ReplCommand::UnloadCategories { categories } => {
                        handle_unload_categories(coordinator, &categories);
                    }
                    ReplCommand::UnloadAll => {
                        handle_unload_all(coordinator);
                    }
                    ReplCommand::Set { key, value } => {
                        handle_set(coordinator, &key, &value);
                    }
                    ReplCommand::Get { key } => {
                        handle_get(coordinator, key.as_deref());
                    }
                    ReplCommand::Clear => {
                        conversation.clear();
                        println!("  conversation cleared.");
                    }
                    ReplCommand::Chat { text } => {
                        handle_chat(&mut *coordinator, &text, &mut conversation);
                    }
                    ReplCommand::Tool { text } => handle_tool(&mut *coordinator, &text),
                    ReplCommand::Talk { text } => handle_talk(coordinator, &text),
                    ReplCommand::Listen { file } => {
                        handle_listen(coordinator, file.as_deref());
                    }
                    ReplCommand::Imagine { prompt } => handle_imagine(coordinator, &prompt),
                    ReplCommand::Image { path, prompt } => {
                        handle_image(&mut *coordinator, &path, &prompt);
                    }
                    ReplCommand::Embed { text } => handle_embed(coordinator, &text),
                    ReplCommand::Unknown(s) => {
                        println!("unknown command: {s}. Type /help for commands.");
                    }
                    ReplCommand::Ping => {
                        println!("pong");
                    }
                }
                if needs_refresh {
                    refresh(&mut rl, coordinator);
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    coordinator.unload_all();
                    println!("goodbye (interrupted).");
                    break;
                }
                println!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                coordinator.unload_all();
                break;
            }
            Err(e) => {
                tracing::error!("readline error: {e}");
                break;
            }
        }
    }

    Ok(())
}

// ── Help ────────────────────────────────────────────────────────────────────

fn print_help(command: Option<&str>) {
    match command {
        None => print_help_overview(),
        Some(cmd) => print_help_detail(cmd),
    }
}

fn print_help_overview() {
    println!("Commands:");
    println!("  /chat <text>                     Chat with the loaded chat model");
    println!("  /tool <text>                     Send a tool-calling request");
    println!("  /talk <text>                     Synthesize speech from text");
    println!("  /listen [file]                   Record or transcribe audio");
    println!("  /imagine <prompt>                Generate an image from text");
    println!("  /image <path> <prompt>           Analyze an image with a prompt");
    println!("  /embed <text>                    Generate text embeddings");
    println!("  /load categories <c,c,...>       Load default models for categories");
    println!("  /load models <m,m,...>           Load specific models by name");
    println!("  /unload models <m,m,...>         Unload specific models");
    println!("  /unload categories <c,c,...>     Unload models for categories");
    println!("  /unload all                      Unload all loaded models");
    println!("  /set <key> <value>               Set a configuration value");
    println!("  /get [key]                       Get configuration value(s)");
    println!("  /list [model]                    List models or show model details");
    println!("  /clear                           Clear conversation history");
    println!("  /stats [model]                   Show inference statistics");
    println!("  /ping                            Test REPL responsiveness");
    println!("  /help [command]                  Show help (or help for a command)");
    println!("  /quit                            Exit the REPL");
    println!();
    println!("  Text without / prefix is treated as /chat input.");
}

fn print_help_detail(cmd: &str) {
    let cmd = cmd.strip_prefix('/').unwrap_or(cmd);
    match cmd {
        "chat" => {
            println!("Usage: /chat <text>");
            println!("       <text>          (shorthand — any input without / prefix)");
            println!();
            println!("Send a chat message to the default chat model.");
            println!();
            println!("Examples:");
            println!("  /chat What is Rust?");
            println!("  What is Rust?");
        }
        "tool" => {
            println!("Usage: /tool <text>");
            println!();
            println!("Send a tool-calling request to the default tool model.");
            println!();
            println!("Examples:");
            println!("  /tool search for the weather in Amsterdam");
        }
        "talk" => {
            println!("Usage: /talk <text>");
            println!();
            println!("Synthesize speech from the given text using the default talk model.");
            println!();
            println!("Examples:");
            println!("  /talk Hello, welcome to nexo!");
        }
        "listen" => {
            println!("Usage: /listen [file]");
            println!();
            println!("Without arguments: record from the microphone and transcribe.");
            println!("With a file path: load and transcribe the given audio file.");
            println!();
            println!("Examples:");
            println!("  /listen");
            println!("  /listen recording.mp3");
        }
        "imagine" => {
            println!("Usage: /imagine <prompt>");
            println!();
            println!("Generate an image from the given text prompt.");
            println!();
            println!("Examples:");
            println!("  /imagine a sunset over a mountain lake");
        }
        "image" => {
            println!("Usage: /image <path> <prompt>");
            println!();
            println!("Analyze an image file with a text prompt.");
            println!();
            println!("Examples:");
            println!("  /image photo.jpg What do you see in this image?");
        }
        "embed" => {
            println!("Usage: /embed <text>");
            println!();
            println!("Generate an embedding vector for the given text.");
            println!();
            println!("Examples:");
            println!("  /embed The quick brown fox jumps over the lazy dog");
        }
        "load" => {
            println!("Usage: /load categories <category>,<category>,...");
            println!("       /load models <model>,<model>,...");
            println!();
            println!("Load models into memory for inference.");
            println!("  categories  Load the default model for each given category.");
            println!("  models      Load specific models by name.");
            println!();
            println!("Categories: chat, tool, image, listen, talk, imagine, embed");
            println!();
            println!("Examples:");
            println!("  /load categories chat,talk");
            println!("  /load models parler-mini");
        }
        "unload" => {
            println!("Usage: /unload models <model>,<model>,...");
            println!("       /unload categories <category>,<category>,...");
            println!("       /unload all");
            println!();
            println!("Unload models from memory.");
            println!("  models      Unload specific models by name.");
            println!("  categories  Unload models serving given categories.");
            println!("  all         Unload all loaded models.");
            println!();
            println!("Examples:");
            println!("  /unload models parler-mini");
            println!("  /unload categories talk");
            println!("  /unload all");
        }
        "set" => {
            println!("Usage: /set <key> <value>");
            println!("       /set                    (show all config — same as /get)");
            println!();
            println!("Set a configuration value and save to ~/.nexo/nexo-ai.toml.");
            println!();
            println!("Keys:");
            println!("  default-<category>       Default model for a category");
            println!("  startup-categories       Comma-separated categories to load on start");
            println!(
                "  <model>.<setting>        Per-model setting (temperature, max_tokens, etc.)"
            );
            println!();
            println!("Examples:");
            println!("  /set default-talk parler-mini");
            println!("  /set startup-categories chat,talk");
            println!("  /set parler-mini.voice_description \"warm female voice\"");
        }
        "get" => {
            println!("Usage: /get [key]");
            println!();
            println!("Show a configuration value, or show all configuration if no key given.");
            println!();
            println!("Examples:");
            println!("  /get");
            println!("  /get default-chat");
            println!("  /get startup-categories");
        }
        "list" => {
            println!("Usage: /list [model]");
            println!();
            println!("Without arguments: show a table of all registered models.");
            println!("With a model name: show detailed info about that model.");
            println!();
            println!("Examples:");
            println!("  /list");
            println!("  /list parler-mini");
        }
        "stats" => {
            println!("Usage: /stats [model]");
            println!();
            println!("Show inference performance statistics.");
            println!("Without arguments: show stats for all models.");
            println!("With a model name: show detailed stats for that model.");
            println!();
            println!("Examples:");
            println!("  /stats");
            println!("  /stats parler-mini");
        }
        "clear" => {
            println!("Usage: /clear");
            println!();
            println!("Clear conversation history. Keeps the system prompt if one was set.");
            println!("Use this to start a fresh conversation with the chat model.");
        }
        "quit" | "exit" => {
            println!("Usage: /quit  (or /q, /exit)");
            println!();
            println!("Unload all models and exit the REPL.");
        }
        _ => {
            println!("No help available for '{cmd}'. Type /help for all commands.");
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn category_list_str() -> String {
    ModelCategory::all()
        .iter()
        .map(|c| c.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn settings_pairs(s: &crate::config::ModelSettings) -> Vec<(&'static str, String)> {
    let mut pairs = Vec::new();
    if let Some(v) = s.temperature {
        pairs.push(("temperature", v.to_string()));
    }
    if let Some(v) = s.max_tokens {
        pairs.push(("max_tokens", v.to_string()));
    }
    if let Some(v) = s.top_p {
        pairs.push(("top_p", v.to_string()));
    }
    if let Some(v) = s.seed {
        pairs.push(("seed", v.to_string()));
    }
    if let Some(ref v) = s.voice_description {
        pairs.push(("voice_description", v.clone()));
    }
    pairs
}

// ── Handlers ────────────────────────────────────────────────────────────────

fn handle_list_models(coordinator: &mut Coordinator, model: Option<&str>) {
    match model {
        None => handle_list_all(coordinator),
        Some(name) => handle_list_detail(coordinator, name),
    }
}

fn handle_list_all(coordinator: &mut Coordinator) {
    let models = coordinator.list_models();
    if models.is_empty() {
        println!("  no models registered.");
        return;
    }

    // Build default column: for each model, collect categories it is the default for.
    let defaults_for = |name: &str| -> String {
        ModelCategory::all()
            .iter()
            .filter(|cat| {
                coordinator
                    .active_model_for(**cat)
                    .is_some_and(|default| default == name)
            })
            .map(|cat| cat.as_str())
            .collect::<Vec<_>>()
            .join(",")
    };

    println!(
        "  {:<25} {:<15} {:<15} {:<8} {:<12} {:<10} DESCRIPTION",
        "NAME", "FAMILY", "CATEGORIES", "SIZE", "DOWNLOADED", "DEFAULT"
    );
    println!("  {}", "-".repeat(100));

    for model in &models {
        let cats: Vec<&str> = model.categories.iter().map(|c| c.as_str()).collect();
        let downloaded = if model.is_downloaded { "yes" } else { "no" };
        let default = defaults_for(&model.name);
        println!(
            "  {:<25} {:<15} {:<15} {:<8} {:<12} {:<10} {}",
            model.name,
            model.family,
            cats.join(","),
            format!("{:.1}G", model.size_gb),
            downloaded,
            default,
            model.description
        );
    }
}

fn handle_list_detail(coordinator: &mut Coordinator, name: &str) {
    let models = coordinator.list_models();
    let Some(model) = models.iter().find(|m| m.name == name) else {
        println!("  unknown model: '{name}'");
        return;
    };

    let cats: Vec<&str> = model.categories.iter().map(|c| c.as_str()).collect();
    let downloaded = if model.is_downloaded { "yes" } else { "no" };
    let loaded = if model.is_loaded { "yes" } else { "no" };

    println!("  Name:         {}", model.name);
    println!("  Family:       {}", model.family);
    println!("  Categories:   {}", cats.join(", "));
    println!("  Size:         {:.1} GB", model.size_gb);
    println!("  Downloaded:   {downloaded}");
    println!("  Loaded:       {loaded}");

    // Show which categories this model is the default for.
    let default_cats: Vec<&str> = ModelCategory::all()
        .iter()
        .filter(|cat| {
            coordinator
                .active_model_for(**cat)
                .is_some_and(|d| d == model.name)
        })
        .map(|c| c.as_str())
        .collect();
    if !default_cats.is_empty() {
        println!("  Default for:  {}", default_cats.join(", "));
    }

    println!("  Description:  {}", model.description);

    let settings = coordinator.config().model_settings(name);
    let pairs = settings_pairs(&settings);
    if !pairs.is_empty() {
        println!();
        println!("  Settings:");
        for (k, v) in &pairs {
            println!("    {k:<20}{v}");
        }
    }
}

fn handle_load_categories(coordinator: &mut Coordinator, categories: &[String]) {
    let parsed: Vec<ModelCategory> = categories
        .iter()
        .filter_map(|s| s.parse::<ModelCategory>().ok())
        .collect();

    if parsed.is_empty() {
        println!("  no valid categories. Available: {}", category_list_str());
        return;
    }

    if let Err(e) = coordinator.load_active_models(&parsed) {
        println!("  error loading models: {e}");
    }
}

fn handle_load_models(coordinator: &mut Coordinator, models: &[String]) {
    for name in models {
        let manifest = find_manifest(name);

        // For each category of this model, unload any existing active model first.
        if let Some(ref manifest) = manifest {
            let loaded: Vec<(String, Vec<ModelCategory>)> = coordinator
                .loaded_models()
                .iter()
                .map(|(n, cats)| (n.to_string(), cats.to_vec()))
                .collect();
            for cat in &manifest.categories {
                if let Some(current) = coordinator.active_model_for(*cat).map(str::to_string) {
                    if current != *name && coordinator.is_model_loaded(&current) {
                        let still_needed = loaded.iter().any(|(n, cats)| {
                            *n == current
                                && cats.iter().any(|c| {
                                    c != cat
                                        && coordinator.active_model_for(*c).is_some_and(|a| a == current)
                                })
                        });
                        if !still_needed {
                            if let Err(e) = coordinator.unload_model(&current) {
                                println!("  error unloading {current}: {e}");
                            } else {
                                println!("  unloaded {current} ({cat})");
                            }
                        }
                    }
                }
            }
        }

        match coordinator.load_model(name) {
            Ok(()) => {
                println!("  loaded {name}");
                if let Some(ref manifest) = manifest {
                    for cat in &manifest.categories {
                        coordinator.set_active_model(*cat, name.clone());
                    }
                }
            }
            Err(e) => println!("  error loading {name}: {e}"),
        }
    }
}

fn handle_unload_models(coordinator: &mut Coordinator, models: &[String]) {
    for name in models {
        match coordinator.unload_model(name) {
            Ok(()) => {
                println!("  unloaded {name}");
                if let Some(manifest) = find_manifest(name) {
                    for cat in &manifest.categories {
                        if coordinator.active_model_for(*cat).is_some_and(|a| a == name.as_str()) {
                            coordinator.remove_active_model(*cat);
                        }
                    }
                }
            }
            Err(e) => println!("  error unloading {name}: {e}"),
        }
    }
}

fn handle_unload_categories(coordinator: &mut Coordinator, categories: &[String]) {
    let parsed: Vec<ModelCategory> = categories
        .iter()
        .filter_map(|s| s.parse::<ModelCategory>().ok())
        .collect();

    if parsed.is_empty() {
        println!("  no valid categories. Available: {}", category_list_str());
        return;
    }

    let loaded: Vec<(String, Vec<ModelCategory>)> = coordinator
        .loaded_models()
        .iter()
        .map(|(n, cats)| (n.to_string(), cats.to_vec()))
        .collect();

    for category in &parsed {
        let to_unload: Vec<&str> = loaded
            .iter()
            .filter(|(_, cats)| cats.contains(category))
            .map(|(name, _)| name.as_str())
            .collect();

        if to_unload.is_empty() {
            println!("  no loaded models for {category}");
        } else {
            for name in to_unload {
                match coordinator.unload_model(&name) {
                    Ok(()) => {
                        println!("  unloaded {name} ({category})");
                        coordinator.remove_active_model(*category);
                    }
                    Err(e) => println!("  error unloading {name}: {e}"),
                }
            }
        }
    }
}

fn handle_unload_all(coordinator: &mut Coordinator) {
    let count = coordinator.loaded_model_count();
    coordinator.unload_all();
    coordinator.clear_active_models();
    println!("  unloaded {count} model(s)");
}

fn handle_set(coordinator: &mut Coordinator, key: &str, value: &str) {
    if let Some(cat_str) = key.strip_prefix("default-") {
        if let Ok(category) = cat_str.parse::<ModelCategory>() {
            coordinator.set_active_model(category, value.to_string());
            println!("  set default-{cat_str} = {value}");
        } else {
            println!("  unknown category: {cat_str}");
        }
    } else if key == "startup-categories" {
        let cats = parse_comma_list(value);
        coordinator.config_mut().startup_categories = cats.clone();
        save_config(coordinator);
        println!("  set startup-categories = {}", cats.join(","));
    } else if let Some((model_name, setting)) = key.split_once('.') {
        let settings = coordinator
            .config_mut()
            .models
            .entry(model_name.to_string())
            .or_default();
        match setting {
            "temperature" => {
                if let Ok(v) = value.parse::<f64>() {
                    settings.temperature = Some(v);
                    save_config(coordinator);
                    println!("  set {key} = {v}");
                } else {
                    println!("  invalid value for temperature: {value}");
                }
            }
            "max_tokens" => {
                if let Ok(v) = value.parse::<usize>() {
                    settings.max_tokens = Some(v);
                    save_config(coordinator);
                    println!("  set {key} = {v}");
                } else {
                    println!("  invalid value for max_tokens: {value}");
                }
            }
            "top_p" => {
                if let Ok(v) = value.parse::<f64>() {
                    settings.top_p = Some(v);
                    save_config(coordinator);
                    println!("  set {key} = {v}");
                } else {
                    println!("  invalid value for top_p: {value}");
                }
            }
            "seed" => {
                if let Ok(v) = value.parse::<u64>() {
                    settings.seed = Some(v);
                    save_config(coordinator);
                    println!("  set {key} = {v}");
                } else {
                    println!("  invalid value for seed: {value}");
                }
            }
            "voice_description" => {
                settings.voice_description = Some(value.to_string());
                save_config(coordinator);
                println!("  set {key} = {value}");
            }
            _ => {
                println!("  unknown setting: {setting}");
                println!("  available: temperature, max_tokens, top_p, seed, voice_description");
            }
        }
    } else {
        println!("  unknown key: {key}");
        println!("  use: default-<category>, startup-categories, or <model>.<setting>");
    }
}

fn handle_get(coordinator: &Coordinator, key: Option<&str>) {
    let config = coordinator.config();
    match key {
        None => {
            // Show full config.
            println!(
                "  startup-categories: {}",
                config.startup_categories.join(",")
            );
            println!();
            println!("  active models:");
            if config.active_models.is_empty() {
                println!("    (none)");
            } else {
                let mut active: Vec<_> = config.active_models.iter().collect();
                active.sort_by_key(|(k, _)| k.as_str());
                for (cat, model) in active {
                    println!("    default-{cat} = {model}");
                }
            }
            if !config.models.is_empty() {
                println!();
                println!("  model settings:");
                let mut model_names: Vec<_> = config.models.keys().collect();
                model_names.sort();
                for name in model_names {
                    let s = config.model_settings(name);
                    println!("    [{name}]");
                    for (k, v) in settings_pairs(&s) {
                        println!("      {k} = {v}");
                    }
                }
            }
        }
        Some(key) => {
            if let Some(cat_str) = key.strip_prefix("default-") {
                if let Ok(category) = cat_str.parse::<ModelCategory>() {
                    match config.active_model_for(category) {
                        Some(model) => println!("  default-{cat_str} = {model}"),
                        None => println!("  default-{cat_str} is not set"),
                    }
                } else {
                    println!("  unknown category: {cat_str}");
                }
            } else if key == "startup-categories" {
                println!(
                    "  startup-categories = {}",
                    config.startup_categories.join(",")
                );
            } else if let Some((model_name, setting)) = key.split_once('.') {
                let s = config.model_settings(model_name);
                match setting {
                    "temperature" => print_opt(key, s.temperature),
                    "max_tokens" => print_opt(key, s.max_tokens),
                    "top_p" => print_opt(key, s.top_p),
                    "seed" => print_opt(key, s.seed),
                    "voice_description" => print_opt(key, s.voice_description.as_deref()),
                    _ => println!("  unknown setting: {setting}"),
                }
            } else {
                println!("  unknown key: {key}");
            }
        }
    }
}

fn print_opt<T: std::fmt::Display>(key: &str, value: Option<T>) {
    match value {
        Some(v) => println!("  {key} = {v}"),
        None => println!("  {key} is not set"),
    }
}

fn save_config(coordinator: &Coordinator) {
    if let Err(e) = coordinator.config().save() {
        println!("  warning: failed to save config: {e}");
    }
}


fn print_token_stats(tokens_generated: usize, inference_time_ms: u64) {
    let secs = inference_time_ms as f64 / 1000.0;
    let tok_s = if secs > 0.0 {
        tokens_generated as f64 / secs
    } else {
        0.0
    };
    println!(
        "\n  ({} tokens in {:.1}s, {:.1} tok/s)",
        tokens_generated, secs, tok_s,
    );
}

const DEFAULT_CHAT_MAX_TOKENS: usize = 2048;
const DEFAULT_CHAT_TEMPERATURE: f64 = 0.7;
const DEFAULT_CHAT_TOP_P: f64 = 0.9;

fn handle_chat(coordinator: &mut Coordinator, text: &str, conversation: &mut ConversationManager) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Chat)
        .map(str::to_string)
    else {
        println!("  no chat model loaded. Use /load chat");
        return;
    };

    conversation.push(ChatMessage {
        role: ChatRole::User,
        content: text.to_string(),
    });

    let settings = coordinator.config().model_settings(&model_name);
    let request = crate::shared::types::ChatRequest {
        messages: conversation.messages().to_vec(),
        max_tokens: settings.max_tokens.unwrap_or(DEFAULT_CHAT_MAX_TOKENS),
        temperature: settings.temperature.unwrap_or(DEFAULT_CHAT_TEMPERATURE),
        top_p: settings.top_p.unwrap_or(DEFAULT_CHAT_TOP_P),
    };

    println!("  [chat via {model_name}] thinking...");

    let result = {
        let model = coordinator.model_mut(&model_name);
        let chat = model.and_then(|m| m.as_chat());
        let Some(chat) = chat else {
            println!("  error: model '{model_name}' does not support chat");
            return;
        };
        chat.chat(&request)
    };

    match result {
        Ok(response) => {
            conversation.push(ChatMessage {
                role: ChatRole::Assistant,
                content: response.text.clone(),
            });

            println!();
            println!("  {}", response.text);
            print_token_stats(response.tokens_generated, response.inference_time_ms);
            coordinator.stats_mut().record_text_generation(
                &model_name,
                ModelCategory::Chat,
                response.tokens_generated,
                response.inference_time_ms,
                None,
                None,
            );
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_tool(coordinator: &mut Coordinator, text: &str) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Tool)
        .map(str::to_string)
    else {
        println!("  no tool model loaded. Use /load tool");
        return;
    };

    let settings = coordinator.config().model_settings(&model_name);
    let request = crate::shared::types::ToolCallRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: text.to_string(),
        }],
        tools: vec![],
        max_tokens: settings.max_tokens.unwrap_or(DEFAULT_CHAT_MAX_TOKENS),
        temperature: settings.temperature.unwrap_or(0.3),
    };

    println!("  [tool via {model_name}] thinking...");

    let result = {
        let model = coordinator.model_mut(&model_name);
        let tool = model.and_then(|m| m.as_tool());
        let Some(tool) = tool else {
            println!("  error: model '{model_name}' does not support tool calling");
            return;
        };
        tool.call_tools(&request)
    };

    match result {
        Ok(response) => {
            if let Some(reasoning) = &response.reasoning {
                println!("\n  reasoning: {reasoning}");
            }
            if response.tool_calls.is_empty() {
                println!("  (no tool calls produced)");
            } else {
                for tc in &response.tool_calls {
                    println!("  tool call: {} {}", tc.name, tc.arguments);
                }
            }
            let secs = response.inference_time_ms as f64 / 1000.0;
            println!(
                "\n  ({} tokens in {:.1}s)",
                response.tokens_generated, secs,
            );
            coordinator.stats_mut().record_text_generation(
                &model_name,
                ModelCategory::Tool,
                response.tokens_generated,
                response.inference_time_ms,
                None,
                None,
            );
        }
        Err(e) => println!("  error: {e}"),
    }
}

const DEFAULT_VOICE_DESCRIPTION: &str = "A clear female speaker with a warm tone.";
const DEFAULT_TALK_MAX_TOKENS: usize = 2048;
const DEFAULT_TALK_TEMPERATURE: f64 = 1.0;
const DEFAULT_TALK_SEED: u64 = 0;

fn handle_talk(coordinator: &mut Coordinator, text: &str) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Talk)
        .map(str::to_string)
    else {
        println!("  no talk model loaded. Use /load talk");
        return;
    };

    let settings = coordinator.config().model_settings(&model_name);
    let request = crate::shared::types::TalkRequest {
        text: text.to_string(),
        voice_description: settings
            .voice_description
            .unwrap_or_else(|| DEFAULT_VOICE_DESCRIPTION.to_string()),
        max_tokens: settings.max_tokens.unwrap_or(DEFAULT_TALK_MAX_TOKENS),
        temperature: settings.temperature.unwrap_or(DEFAULT_TALK_TEMPERATURE),
        seed: settings.seed.unwrap_or(DEFAULT_TALK_SEED),
    };

    println!("  [talk via {model_name}] synthesizing...");

    let result = {
        let model = coordinator.model_mut(&model_name);
        let talk = model.and_then(|m| m.as_talk());
        let Some(talk) = talk else {
            println!("  error: model '{model_name}' does not support talk");
            return;
        };
        talk.synthesize(&request)
    };

    match result {
        Ok(response) => {
            let samples_generated = response.pcm_samples.len();
            let sample_rate = response.sample_rate;
            let inference_time_ms = response.inference_time_ms;
            let buffer =
                crate::audio::AudioBuffer::new(response.pcm_samples, response.sample_rate, 1);
            println!(
                "  generated {:.1}s of audio in {:.1}s",
                buffer.duration_secs(),
                inference_time_ms as f64 / 1000.0,
            );
            coordinator.stats_mut().record_synthesis(
                &model_name,
                samples_generated,
                sample_rate,
                inference_time_ms,
            );
            if let Err(e) = crate::audio::play(&buffer) {
                println!("  error playing audio: {e}");
            }
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_listen(coordinator: &mut Coordinator, file: Option<&str>) {
    let audio = match file {
        Some(path) => {
            println!("  loading audio from {path}...");
            match crate::audio::load_file(std::path::Path::new(path)) {
                Ok(buf) => buf.to_mono(),
                Err(e) => {
                    println!("  error loading audio file: {e}");
                    return;
                }
            }
        }
        None => {
            println!("  recording from microphone (silence stops recording)...");
            match crate::audio::record_microphone(&crate::audio::RecordConfig::default()) {
                Ok(buf) => buf,
                Err(e) => {
                    println!("  error recording: {e}");
                    return;
                }
            }
        }
    };

    println!(
        "  audio: {:.1}s at {} Hz",
        audio.duration_secs(),
        audio.sample_rate,
    );

    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Listen)
        .map(str::to_string)
    else {
        println!("  no listen model loaded. Use /load listen");
        return;
    };

    println!("  [listen via {model_name}] transcribing...");

    let audio_duration_ms = (audio.duration_secs() * 1000.0) as u64;
    let request = crate::shared::types::ListenRequest {
        pcm_samples: audio.samples,
        sample_rate: audio.sample_rate,
        language: None,
    };

    let result = {
        let model = coordinator.model_mut(&model_name);
        let listen = model.and_then(|m| m.as_listen());
        let Some(listen) = listen else {
            println!("  error: model '{model_name}' does not support listen");
            return;
        };
        listen.transcribe(&request)
    };

    match result {
        Ok(response) => {
            println!();
            println!("  {}", response.text);
            if let Some(lang) = &response.language {
                println!("  language: {lang}");
            }
            println!("  ({:.1}s)", response.inference_time_ms as f64 / 1000.0,);
            coordinator.stats_mut().record_transcription(
                &model_name,
                audio_duration_ms,
                response.inference_time_ms,
            );
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_imagine(coordinator: &mut Coordinator, prompt: &str) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Imagine)
        .map(str::to_string)
    else {
        println!("  no imagine model loaded. Use /load imagine");
        return;
    };

    let settings = coordinator.config().model_settings(&model_name);

    // Derive sensible defaults from model family rather than coupling to a specific model type.
    let (family_steps, family_guidance) = find_manifest(&model_name)
        .map(|m| match m.manifest.family.as_str() {
            "flux" => (4u32, 0.0f64),
            _ => (20, 7.5),
        })
        .unwrap_or((20, 7.5));

    let request = crate::shared::types::ImagineRequest {
        prompt: prompt.to_string(),
        width: settings.default_width.unwrap_or(512),
        height: settings.default_height.unwrap_or(512),
        steps: settings.default_steps.unwrap_or(family_steps),
        guidance: settings.default_guidance.unwrap_or(family_guidance),
        seed: settings.seed.unwrap_or(0),
        batch_size: 1,
    };

    println!(
        "  [imagine via {model_name}] generating {}x{} ({} steps)...",
        request.width, request.height, request.steps
    );

    let result = {
        let model = coordinator.model_mut(&model_name);
        let imagine = model.and_then(|m| m.as_imagine());
        let Some(imagine) = imagine else {
            println!("  error: model '{model_name}' does not support imagine");
            return;
        };
        imagine.imagine(&request)
    };

    match result {
        Ok(response) => {
            println!(
                "  generated {} image(s) in {:.1}s (seed={})",
                response.images.len(),
                response.inference_time_ms as f64 / 1000.0,
                response.seed_used,
            );
            let total_pixels = response
                .images
                .iter()
                .map(|i| i.width as u64 * i.height as u64)
                .sum();
            coordinator.stats_mut().record_image_generation(
                &model_name,
                response.images.len() as u32,
                request.steps,
                total_pixels,
                response.inference_time_ms,
            );
            for img in &response.images {
                let output_dir = std::path::PathBuf::from("datasets/images/generated");
                if let Err(e) = std::fs::create_dir_all(&output_dir) {
                    println!("  error creating output dir: {e}");
                    continue;
                }
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let path = output_dir.join(format!("{timestamp}_{}.png", response.seed_used));
                if let Err(e) = std::fs::write(&path, &img.data) {
                    println!("  error saving image: {e}");
                } else {
                    println!("  saved to {}", path.display());
                }
            }
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_image(coordinator: &mut Coordinator, path: &str, prompt: &str) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Image)
        .map(str::to_string)
    else {
        println!("  no image model loaded. Use /load image");
        return;
    };

    let image_data = match std::fs::read(path) {
        Ok(data) => data,
        Err(e) => {
            println!("  error reading image file: {e}");
            return;
        }
    };

    let settings = coordinator.config().model_settings(&model_name);
    let request = crate::shared::types::ImageAnalysisRequest {
        image_data,
        prompt: prompt.to_string(),
        max_tokens: settings.max_tokens.unwrap_or(DEFAULT_CHAT_MAX_TOKENS),
        temperature: settings.temperature.unwrap_or(DEFAULT_CHAT_TEMPERATURE),
    };

    println!("  [image via {model_name}] analyzing...");

    let result = {
        let model = coordinator.model_mut(&model_name);
        let image = model.and_then(|m| m.as_image());
        let Some(image) = image else {
            println!("  error: model '{model_name}' does not support image analysis");
            return;
        };
        image.analyze_image(&request)
    };

    match result {
        Ok(response) => {
            println!();
            println!("  {}", response.text);
            print_token_stats(response.tokens_generated, response.inference_time_ms);
            coordinator.stats_mut().record_text_generation(
                &model_name,
                ModelCategory::Image,
                response.tokens_generated,
                response.inference_time_ms,
                None,
                None,
            );
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_embed(coordinator: &mut Coordinator, text: &str) {
    let Some(model_name) = coordinator
        .active_model_for(ModelCategory::Embed)
        .map(str::to_string)
    else {
        println!("  no embed model loaded. Use /load embed");
        return;
    };

    let request = crate::shared::types::EmbedRequest {
        texts: vec![text.to_string()],
    };

    println!("  [embed via {model_name}] computing...");

    let result = {
        let model = coordinator.model_mut(&model_name);
        let embed = model.and_then(|m| m.as_embed());
        let Some(embed) = embed else {
            println!("  error: model '{model_name}' does not support embed");
            return;
        };
        embed.embed(&request)
    };

    match result {
        Ok(response) => {
            if let Some(emb) = response.embeddings.first() {
                let preview: Vec<String> = emb.iter().take(8).map(|v| format!("{v:.4}")).collect();
                let suffix = if emb.len() > 8 {
                    format!(", ... ({} dims)", emb.len())
                } else {
                    String::new()
                };
                println!("  [{}{}]", preview.join(", "), suffix);
            }
            println!("  ({:.1}s)", response.inference_time_ms as f64 / 1000.0);
        }
        Err(e) => println!("  error: {e}"),
    }
}

fn handle_stats(coordinator: &Coordinator, model: Option<&str>) {
    match model {
        None => {
            let all = coordinator.stats().all_stats();
            stats_display::print_stats_table(&all);
        }
        Some(name) => {
            let stats: Vec<_> = ModelCategory::all()
                .iter()
                .filter_map(|cat| coordinator.stats().model_stats(name, *cat))
                .collect();
            let lifecycle: Vec<_> = coordinator.stats().lifecycle_history(name);
            stats_display::print_model_detail(&stats, &lifecycle);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty() {
        assert_eq!(parse_repl_input(""), ReplCommand::Empty);
        assert_eq!(parse_repl_input("   "), ReplCommand::Empty);
    }

    #[test]
    fn parse_plain_text_as_chat() {
        assert_eq!(
            parse_repl_input("hello world"),
            ReplCommand::Chat {
                text: "hello world".to_string()
            }
        );
    }

    #[test]
    fn parse_chat_command() {
        assert_eq!(
            parse_repl_input("/chat what is rust?"),
            ReplCommand::Chat {
                text: "what is rust?".to_string()
            }
        );
    }

    #[test]
    fn parse_tool_command() {
        assert_eq!(
            parse_repl_input("/tool generate_image --prompt cats"),
            ReplCommand::Tool {
                text: "generate_image --prompt cats".to_string()
            }
        );
    }

    #[test]
    fn parse_talk_command() {
        assert_eq!(
            parse_repl_input("/talk tell me a story"),
            ReplCommand::Talk {
                text: "tell me a story".to_string()
            }
        );
    }

    #[test]
    fn parse_listen_no_args() {
        assert_eq!(
            parse_repl_input("/listen"),
            ReplCommand::Listen { file: None }
        );
    }

    #[test]
    fn parse_listen_with_file() {
        assert_eq!(
            parse_repl_input("/listen recording.mp3"),
            ReplCommand::Listen {
                file: Some("recording.mp3".to_string())
            }
        );
    }

    #[test]
    fn parse_imagine() {
        assert_eq!(
            parse_repl_input("/imagine a cute cat"),
            ReplCommand::Imagine {
                prompt: "a cute cat".to_string()
            }
        );
    }

    #[test]
    fn parse_load_categories() {
        assert_eq!(
            parse_repl_input("/load categories chat,tool"),
            ReplCommand::LoadCategories {
                categories: vec!["chat".to_string(), "tool".to_string()]
            }
        );
    }

    #[test]
    fn parse_load_models() {
        assert_eq!(
            parse_repl_input("/load models parler-mini,parler-large"),
            ReplCommand::LoadModels {
                models: vec!["parler-mini".to_string(), "parler-large".to_string()]
            }
        );
    }

    #[test]
    fn parse_load_bare_categories() {
        // Bare args (no subcommand) → treated as categories for backwards compat.
        assert_eq!(
            parse_repl_input("/load chat,tool"),
            ReplCommand::LoadCategories {
                categories: vec!["chat".to_string(), "tool".to_string()]
            }
        );
    }

    #[test]
    fn parse_unload_models() {
        assert_eq!(
            parse_repl_input("/unload models parler-mini"),
            ReplCommand::UnloadModels {
                models: vec!["parler-mini".to_string()]
            }
        );
    }

    #[test]
    fn parse_unload_categories() {
        assert_eq!(
            parse_repl_input("/unload categories talk,chat"),
            ReplCommand::UnloadCategories {
                categories: vec!["talk".to_string(), "chat".to_string()]
            }
        );
    }

    #[test]
    fn parse_unload_all() {
        assert_eq!(parse_repl_input("/unload all"), ReplCommand::UnloadAll);
    }

    #[test]
    fn parse_set_command() {
        assert_eq!(
            parse_repl_input("/set default-chat qwen3-8b"),
            ReplCommand::Set {
                key: "default-chat".to_string(),
                value: "qwen3-8b".to_string()
            }
        );
    }

    #[test]
    fn parse_set_strips_quotes() {
        assert_eq!(
            parse_repl_input("/set default-chat \"qwen3-8b\""),
            ReplCommand::Set {
                key: "default-chat".to_string(),
                value: "qwen3-8b".to_string()
            }
        );
    }

    #[test]
    fn parse_set_no_args_shows_config() {
        assert_eq!(parse_repl_input("/set"), ReplCommand::Get { key: None });
    }

    #[test]
    fn parse_set_single_arg_shows_value() {
        assert_eq!(
            parse_repl_input("/set default-chat"),
            ReplCommand::Get {
                key: Some("default-chat".to_string())
            }
        );
    }

    #[test]
    fn parse_get_no_args() {
        assert_eq!(parse_repl_input("/get"), ReplCommand::Get { key: None });
    }

    #[test]
    fn parse_get_with_key() {
        assert_eq!(
            parse_repl_input("/get default-talk"),
            ReplCommand::Get {
                key: Some("default-talk".to_string())
            }
        );
    }

    #[test]
    fn parse_list_no_args() {
        assert_eq!(
            parse_repl_input("/list"),
            ReplCommand::ListModels { model: None }
        );
    }

    #[test]
    fn parse_list_with_model() {
        assert_eq!(
            parse_repl_input("/list parler-mini"),
            ReplCommand::ListModels {
                model: Some("parler-mini".to_string())
            }
        );
    }

    #[test]
    fn parse_help_no_args() {
        assert_eq!(
            parse_repl_input("/help"),
            ReplCommand::Help { command: None }
        );
        assert_eq!(parse_repl_input("/h"), ReplCommand::Help { command: None });
        assert_eq!(parse_repl_input("/?"), ReplCommand::Help { command: None });
    }

    #[test]
    fn parse_help_with_command() {
        assert_eq!(
            parse_repl_input("/help load"),
            ReplCommand::Help {
                command: Some("load".to_string())
            }
        );
    }

    #[test]
    fn parse_quit() {
        assert_eq!(parse_repl_input("/quit"), ReplCommand::Quit);
        assert_eq!(parse_repl_input("/q"), ReplCommand::Quit);
        assert_eq!(parse_repl_input("/exit"), ReplCommand::Quit);
    }

    #[test]
    fn parse_stats_no_arg() {
        assert_eq!(
            parse_repl_input("/stats"),
            ReplCommand::Stats { model: None }
        );
    }

    #[test]
    fn parse_stats_with_model() {
        assert_eq!(
            parse_repl_input("/stats qwen3-8b"),
            ReplCommand::Stats {
                model: Some("qwen3-8b".to_string())
            }
        );
    }

    #[test]
    fn parse_unknown() {
        assert_eq!(
            parse_repl_input("/foobar"),
            ReplCommand::Unknown("/foobar".to_string())
        );
    }

    #[test]
    fn parse_image_command() {
        assert_eq!(
            parse_repl_input("/image photo.jpg what do you see?"),
            ReplCommand::Image {
                path: "photo.jpg".to_string(),
                prompt: "what do you see?".to_string()
            }
        );
    }

    // ── Completion tests ────────────────────────────────────────────────

    fn test_data() -> CompletionData {
        CompletionData {
            model_names: vec!["parler-mini".to_string(), "parler-large".to_string()],
            category_names: vec![
                "chat".to_string(),
                "tool".to_string(),
                "image".to_string(),
                "listen".to_string(),
                "talk".to_string(),
                "imagine".to_string(),
                "embed".to_string(),
            ],
            config_keys: vec![
                "default-chat".to_string(),
                "default-tool".to_string(),
                "default-talk".to_string(),
                "startup-categories".to_string(),
            ],
        }
    }

    #[test]
    fn complete_command_prefix() {
        let data = test_data();
        let (start, candidates) = complete_repl("/li", &data);
        assert_eq!(start, 0);
        assert_eq!(candidates.len(), 2); // /list, /listen
    }

    #[test]
    fn complete_list_model_names() {
        let data = test_data();
        let (_, candidates) = complete_repl("/list ", &data);
        assert_eq!(candidates.len(), 2); // parler-mini, parler-large
    }

    #[test]
    fn complete_list_model_partial() {
        let data = test_data();
        let (_, candidates) = complete_repl("/list parler-m", &data);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].display, "parler-mini");
    }

    #[test]
    fn complete_load_subcommands() {
        let data = test_data();
        let (_, candidates) = complete_repl("/load ", &data);
        assert_eq!(candidates.len(), 2); // categories, models
    }

    #[test]
    fn complete_unload_subcommands() {
        let data = test_data();
        let (_, candidates) = complete_repl("/unload ", &data);
        assert_eq!(candidates.len(), 3); // models, categories, all
    }

    #[test]
    fn complete_set_config_keys() {
        let data = test_data();
        let (_, candidates) = complete_repl("/set default-", &data);
        assert_eq!(candidates.len(), 3); // default-chat, default-tool, default-talk
    }

    #[test]
    fn complete_help_topics() {
        let data = test_data();
        let (_, candidates) = complete_repl("/help s", &data);
        let displays: Vec<&str> = candidates.iter().map(|c| c.display.as_str()).collect();
        assert!(displays.contains(&"set"));
        assert!(displays.contains(&"stats"));
    }

    #[test]
    fn complete_plain_text_no_completions() {
        let data = test_data();
        let (_, candidates) = complete_repl("hello", &data);
        assert!(candidates.is_empty());
    }
}
