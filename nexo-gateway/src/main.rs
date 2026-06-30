//! Binary entry point for the NEXO gateway executable.

use std::error::Error as StdError;
use std::process::ExitCode;

use clap::Parser;
use nexo_gateway::cli::base::Cli;

#[tokio::main]
async fn main() -> Result<ExitCode, Box<dyn StdError>> {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    let mut context = cli.common.command_context()?;
    let exit_code = nexo_gateway::cli::base::dispatch(cli.command, &mut context).await?;
    Ok(exit_code)
}
