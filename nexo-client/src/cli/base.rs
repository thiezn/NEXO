use clap::{Parser, Subcommand};
use nexo_ws_schema::SchemaSection;
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(
    name = "nexo-client",
    about = "NEXO Client - Connect to a NEXO Gateway"
)]
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
    /// Start the interactive NEXO terminal UI
    Start {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,

        /// Resume an existing session by ID
        #[arg(long)]
        session: Option<String>,

        /// Session name (used when creating a new session)
        #[arg(long)]
        name: Option<String>,

        /// Model ID to use for inference
        #[arg(long)]
        model: Option<String>,
    },

    /// Generate JSON schemas for the WebSocket protocol
    Schema {
        /// Section to generate
        #[arg(value_enum, default_value_t = SchemaSection::All)]
        section: SchemaSection,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
    },
}
