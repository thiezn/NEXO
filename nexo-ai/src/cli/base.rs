use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(name = "nexo-ai", about = "NEXO AI - Local inference model manager")]
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
    /// Download model weights
    Pull {
        /// Model name, category (chat/tool/image/listen/talk/imagine), or "all"
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if files exist
        #[arg(long)]
        force: bool,
    },

    /// List supported models and their status
    List,

    /// Load default models and start the interactive REPL
    Start {
        /// Categories to load (comma-separated, e.g. "chat,tool")
        #[arg(short, long, value_delimiter = ',')]
        categories: Option<Vec<String>>,
    },
}
