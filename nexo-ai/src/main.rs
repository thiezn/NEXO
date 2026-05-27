//! Command-line entry point for the `nexo-ai` binary.

mod cli;

use std::error::Error as StdError;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use cli_helpers::Runnable;
use cli_helpers::clap::CommonArgs;

#[derive(Parser, Debug)]
#[command(name = "nexo-ai", about = "NEXO local AI runtime tools")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Manage downloaded local models.
    Models(nexo_model_mgmt::ModelsCommand),

    /// Start the standalone local inference TUI.
    Start(cli::StartCommand),
}

fn main() -> Result<ExitCode, Box<dyn StdError>> {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    let mut context = cli.common.command_context()?;
    let exit_code = match cli.command {
        Command::Models(command) => command
            .run(&mut context)
            .map_err(|error| -> Box<dyn StdError> { Box::new(error) })?,
        Command::Start(command) => command
            .run(&mut context)
            .map_err(|error| -> Box<dyn StdError> { Box::new(error) })?,
    };

    Ok(exit_code)
}
