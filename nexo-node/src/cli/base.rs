use crate::cli::commands::{ModelsCommand, init, start};
use nexo_node::Result;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use cli_helpers::CommandContext;
use cli_helpers::clap::CommonArgs;

/// Top-level CLI arguments accepted by the `nexo-node` binary.
#[derive(Parser, Debug)]
#[command(
    name = "nexo-node",
    about = "NEXO Node - Tool capability host and inference service manager"
)]
pub struct Cli {
    #[command(flatten)]
    pub common: CommonArgs,

    #[command(subcommand)]
    pub command: Command,
}

/// Root CLI commands supported by `nexo-node`.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Initialize node configuration
    Init,

    /// Start the node and connect to a gateway
    Start {
        /// Gateway URL (e.g. ws://127.0.0.1:6969)
        #[arg(long)]
        url: Option<String>,
    },

    /// Manage downloaded local models.
    Models(ModelsCommand),
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
pub async fn dispatch(command: Command, context: &mut CommandContext) -> Result<ExitCode> {
    match command {
        Command::Init => {
            init::run()?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Start { url } => {
            start::run(url).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Models(command) => command.run(context).await,
    }
}
