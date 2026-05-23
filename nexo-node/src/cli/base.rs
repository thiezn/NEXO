//! Clap-powered command definitions for the `nexo-node` binary.

use crate::cli::commands::{init, models_list, models_pull, start};
use clap::{Parser, Subcommand};
use cli_helpers::LogLevel;

/// Top-level CLI arguments accepted by the `nexo-node` binary.
#[derive(Parser)]
#[command(
    name = "nexo-node",
    about = "NEXO Node - Tool capability host and inference service manager"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[arg(long, global = true)]
    pub no_color: bool,
}

/// Root CLI commands supported by `nexo-node`.
#[derive(Subcommand)]
pub enum Command {
    /// Initialize node configuration
    Init,

    /// Start the node and connect to a gateway
    Start {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,
    },

    /// Manage local models
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
}

/// Subcommands for local model management.
#[derive(Subcommand)]
pub enum ModelsCommand {
    /// Download a model by name (e.g. "qwen3.5-35b-ab3b") or "all"
    Pull {
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if files already exist
        #[arg(long)]
        force: bool,
    },

    /// List known models and their download status
    List,
}

/// Dispatch a parsed CLI command to its concrete handler.
///
/// # Arguments
///
/// * `command` - The parsed top-level CLI command.
///
/// # Errors
///
/// Returns any error produced by the selected command handler.
pub async fn dispatch(command: Command) -> cli_helpers::Result {
    match command {
        Command::Init => init::run(),
        Command::Start { url } => start::run(url).await,
        Command::Models { action } => match action {
            ModelsCommand::Pull { model, force } => models_pull::run(&model, force).await,
            ModelsCommand::List => models_list::run(),
        },
    }
}
