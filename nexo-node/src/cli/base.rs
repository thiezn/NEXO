use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(name = "nexo-node", about = "NEXO Node - Tool capability host and inference service manager")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[arg(long, global = true)]
    pub no_color: bool,
}

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

    /// Manage local GGUF models
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
}

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
