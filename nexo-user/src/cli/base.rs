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

/// Dispatch a parsed CLI command to its concrete handler.
pub async fn dispatch(command: Command, _context: &mut CommandContext) -> Result<ExitCode> {
    match command {
        Command::Start {
            url,
            session,
            name,
            model,
        } => {
            start::run(start::StartCommand {
                url: url,
                session_id: session,
                session_name: name,
                model_id: model,
            })
            .await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Schema { section, output } => {
            schema::run(section, output.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
