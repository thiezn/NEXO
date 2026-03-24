use clap::{Parser, Subcommand, ValueEnum};
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(name = "nexo-client", about = "NEXO Client - Connect to a NEXO Gateway")]
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
    /// Connect to a NEXO Gateway
    Connect {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,
    },

    /// Generate JSON schemas for the WebSocket protocol
    Schema {
        /// Section to generate
        #[arg(value_enum, default_value_t = SchemaTarget::All)]
        section: SchemaTarget,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SchemaTarget {
    All,
    Frames,
    Connect,
    Methods,
    Events,
    Errors,
}
