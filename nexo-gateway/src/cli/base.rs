use crate::cli::commands::{init, schema, start};
use clap::{Parser, Subcommand};
use cli_helpers::CommandContext;
use cli_helpers::clap::CommonArgs;
use nexo_ws_schema::SchemaSection;
use std::process::ExitCode;

/// Command-line interface for the gateway binary.
#[derive(Parser)]
#[command(name = "nexo", about = "NEXO Gateway - Neural Extension Operator")]
pub struct Cli {
    /// Common CLI flags for tracing and output behavior.
    #[command(flatten)]
    pub common: CommonArgs,

    /// The top-level command to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level commands supported by the gateway binary.
#[derive(Subcommand)]
pub enum Command {
    /// Initialize gateway configuration and database
    Init,

    /// Start the gateway WebSocket server
    Start {
        /// Override bind host
        #[arg(long)]
        host: Option<String>,

        /// Override bind port
        #[arg(long)]
        port: Option<u16>,
    },

    /// Generate JSON schemas for the WebSocket protocol.
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
///
/// # Arguments
///
/// * `command` - The parsed top-level CLI command.
/// * `context` - Shared command I/O and output context.
///
/// # Errors
///
/// Returns any error produced by the selected command handler.
pub async fn dispatch(
    command: Command,
    context: &mut CommandContext,
) -> cli_helpers::Result<ExitCode> {
    let _ = context;
    match command {
        Command::Init => {
            init::run().await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Start { host, port } => {
            start::run(host, port)
                .await
                .map_err(|error| cli_helpers::Error::Other(error.to_string()))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Schema { section, output } => {
            schema::run(section, output).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
