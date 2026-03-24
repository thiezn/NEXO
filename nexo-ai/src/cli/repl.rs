use crate::coordinator::Coordinator;
use crate::shared::types::ModelCategory;
use anyhow::Result;

#[derive(Debug, PartialEq)]
pub enum ReplCommand {
    Chat { text: String },
    Tool { text: String },
    Talk { text: String },
    Listen,
    Imagine { prompt: String },
    Image { path: String, prompt: String },
    StartModels { categories: Vec<String> },
    Config { key: String, value: String },
    ListModels,
    Help,
    Quit,
    Empty,
    Unknown(String),
    Ping,
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
        "/listen" => ReplCommand::Listen,
        "/imagine" => ReplCommand::Imagine {
            prompt: args.to_string(),
        },
        "/image" => {
            // /image <path> <prompt>
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
        "/start" => {
            let cats_str = args.strip_prefix("models ").unwrap_or(args);
            let categories: Vec<String> = cats_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            ReplCommand::StartModels { categories }
        }
        "/config" => {
            let config_parts: Vec<&str> = args.splitn(2, ' ').collect();
            if config_parts.len() == 2 {
                ReplCommand::Config {
                    key: config_parts[0].to_string(),
                    value: config_parts[1].trim_matches('"').to_string(),
                }
            } else {
                ReplCommand::Unknown(input.to_string())
            }
        }
        "/list" => ReplCommand::ListModels,
        "/help" | "/h" | "/?" => ReplCommand::Help,
        "/quit" | "/q" | "/exit" => ReplCommand::Quit,
        _ => ReplCommand::Unknown(input.to_string()),
    }
}

pub fn run_repl(coordinator: &mut Coordinator) -> Result<()> {
    let mut rl = rustyline::DefaultEditor::new()
        .map_err(|e| anyhow::anyhow!("failed to initialize REPL: {e}"))?;

    println!("nexo-ai REPL. Type /help for commands, /quit to exit.");
    println!();

    loop {
        let readline = rl.readline("nexo> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                match parse_repl_input(&line) {
                    ReplCommand::Quit => {
                        coordinator.unload_all();
                        println!("goodbye.");
                        break;
                    }
                    ReplCommand::Help => print_help(),
                    ReplCommand::Empty => {}
                    ReplCommand::ListModels => handle_list_models(coordinator),
                    ReplCommand::StartModels { categories } => {
                        handle_start_models(coordinator, &categories);
                    }
                    ReplCommand::Config { key, value } => {
                        handle_config(coordinator, &key, &value);
                    }
                    ReplCommand::Chat { text } => handle_chat(coordinator, &text),
                    ReplCommand::Tool { text } => handle_tool(coordinator, &text),
                    ReplCommand::Talk { text } => handle_talk(coordinator, &text),
                    ReplCommand::Listen => handle_listen(coordinator),
                    ReplCommand::Imagine { prompt } => handle_imagine(coordinator, &prompt),
                    ReplCommand::Image { path, prompt } => {
                        handle_image(coordinator, &path, &prompt);
                    }
                    ReplCommand::Unknown(s) => {
                        println!("unknown command: {s}. Type /help for commands.");
                    }
                    ReplCommand::Ping => {
                        println!("pong");
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
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

fn print_help() {
    println!("Commands:");
    println!("  /chat <text>            Chat with the loaded chat model");
    println!("  /tool <text>            Send a tool-calling request");
    println!("  /talk <text>            Synthesize speech from text");
    println!("  /listen                 Record and transcribe audio");
    println!("  /imagine <prompt>       Generate an image from text");
    println!("  /image <path> <prompt>  Analyze an image with a prompt");
    println!("  /start models <c,c>     Load models for categories");
    println!("  /config <key> <value>   Change a config setting");
    println!("  /list models            Show loaded/available models");
    println!("  /ping                   Test command to check if REPL is responsive");
    println!("  /help                   Show this help");
    println!("  /quit                   Exit the REPL");
    println!();
    println!("  Text without / prefix is treated as /chat input.");
}

fn handle_list_models(coordinator: &mut Coordinator) {
    let loaded = coordinator.loaded_models();
    if loaded.is_empty() {
        println!("  no models loaded");
    } else {
        for (name, cats) in loaded {
            let cat_str: Vec<&str> = cats.iter().map(|c| c.as_str()).collect();
            println!("  {} [{}]", name, cat_str.join(", "));
        }
    }
    // Also show available but not loaded.
    let all_models = coordinator.list_models();
    let not_loaded: Vec<_> = all_models.iter().filter(|m| !m.is_loaded).collect();
    if !not_loaded.is_empty() {
        println!();
        println!("  available (not loaded):");
        for m in not_loaded {
            let cats: Vec<&str> = m.categories.iter().map(|c| c.as_str()).collect();
            let dl = if m.is_downloaded {
                "downloaded"
            } else {
                "not downloaded"
            };
            println!(
                "    {} [{}] ({:.1}G, {})",
                m.name,
                cats.join(","),
                m.size_gb,
                dl
            );
        }
    }
}

fn handle_start_models(coordinator: &mut Coordinator, categories: &[String]) {
    let parsed: Vec<ModelCategory> = categories
        .iter()
        .filter_map(|s| {
            ModelCategory::all()
                .iter()
                .find(|c| c.as_str() == s)
                .copied()
        })
        .collect();
    if parsed.is_empty() {
        println!("  no valid categories. Available: chat, tool, image, listen, talk, imagine");
        return;
    }
    if let Err(e) = coordinator.load_defaults(&parsed) {
        println!("  error loading models: {e}");
    }
}

fn handle_config(coordinator: &mut Coordinator, key: &str, value: &str) {
    // Handle config keys like "default-chat", "default-tool", etc.
    if let Some(cat_str) = key.strip_prefix("default-") {
        if let Some(category) = ModelCategory::all().iter().find(|c| c.as_str() == cat_str) {
            coordinator.set_default(*category, value.to_string());
            coordinator
                .config_mut()
                .set_default(*category, value.to_string());
            println!("  set default {} model to '{}'", cat_str, value);
        } else {
            println!("  unknown category: {cat_str}");
        }
    } else {
        println!("  unknown config key: {key}. Use 'default-<category>' (e.g. default-chat)");
    }
}

fn dispatch_model(coordinator: &Coordinator, category: ModelCategory, context: &str) {
    if let Some(model_name) = coordinator.default_for(category) {
        if context.is_empty() {
            println!("  [{} via {}]", category, model_name);
        } else {
            println!("  [{} via {}] {}", category, model_name, context);
        }
        println!("  (model dispatch not yet implemented)");
    } else {
        println!(
            "  no {} model loaded. Use /start models {}",
            category, category
        );
    }
}

fn handle_chat(coordinator: &Coordinator, text: &str) {
    dispatch_model(coordinator, ModelCategory::Chat, text);
}

fn handle_tool(coordinator: &Coordinator, text: &str) {
    dispatch_model(coordinator, ModelCategory::Tool, text);
}

fn handle_talk(coordinator: &Coordinator, text: &str) {
    dispatch_model(coordinator, ModelCategory::Talk, text);
}

fn handle_listen(coordinator: &Coordinator) {
    dispatch_model(coordinator, ModelCategory::Listen, "");
}

fn handle_imagine(coordinator: &Coordinator, prompt: &str) {
    dispatch_model(coordinator, ModelCategory::Imagine, prompt);
}

fn handle_image(coordinator: &Coordinator, path: &str, prompt: &str) {
    dispatch_model(
        coordinator,
        ModelCategory::Image,
        &format!("{} - {}", path, prompt),
    );
}

// Unit tests for parse_repl_input
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
    fn parse_listen() {
        assert_eq!(parse_repl_input("/listen"), ReplCommand::Listen);
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
    fn parse_start_models() {
        assert_eq!(
            parse_repl_input("/start models chat,tool"),
            ReplCommand::StartModels {
                categories: vec!["chat".to_string(), "tool".to_string()]
            }
        );
    }

    #[test]
    fn parse_config() {
        assert_eq!(
            parse_repl_input("/config default-chat \"qwen3.5-9b\""),
            ReplCommand::Config {
                key: "default-chat".to_string(),
                value: "qwen3.5-9b".to_string()
            }
        );
    }

    #[test]
    fn parse_list() {
        assert_eq!(parse_repl_input("/list"), ReplCommand::ListModels);
        assert_eq!(parse_repl_input("/list models"), ReplCommand::ListModels);
    }

    #[test]
    fn parse_help() {
        assert_eq!(parse_repl_input("/help"), ReplCommand::Help);
        assert_eq!(parse_repl_input("/h"), ReplCommand::Help);
        assert_eq!(parse_repl_input("/?"), ReplCommand::Help);
    }

    #[test]
    fn parse_quit() {
        assert_eq!(parse_repl_input("/quit"), ReplCommand::Quit);
        assert_eq!(parse_repl_input("/q"), ReplCommand::Quit);
        assert_eq!(parse_repl_input("/exit"), ReplCommand::Quit);
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
}
