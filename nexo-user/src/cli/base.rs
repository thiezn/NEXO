use crate::cli::commands::{schema, start};
use clap::{Parser, Subcommand};
use cli_helpers::CommandContext;
use cli_helpers::clap::CommonArgs;
use nexo_user::Result;
use nexo_ws_schema::SchemaSection;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "nexo-user", about = "NEXO User - Connect to a NEXO Gateway")]
pub struct Cli {
    #[command(flatten)]
    pub common: CommonArgs,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the interactive NEXO terminal UI
    Start {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,
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

/// Dispatch a parsed CLI command to its concrete handler.
pub async fn dispatch(command: Command, _context: &mut CommandContext) -> Result<ExitCode> {
    match command {
        Command::Start { url } => {
            start::run(start::StartCommand { url: url }).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Schema { section, output } => {
            schema::run(section, output.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
