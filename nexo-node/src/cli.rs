use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(name = "nexo-node", about = "NEXO Node - Tool capability host")]
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
    /// Start the node and connect to a gateway
    Start {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,
    },

    /// Initialize node configuration
    Init,
}
