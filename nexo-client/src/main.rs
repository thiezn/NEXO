mod audio;
mod cli;
mod tui;
mod vision;

use clap::Parser;
use cli::base::Cli;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> cli_helpers::Result<ExitCode> {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    let mut context = cli.common.command_context()?;
    cli::base::dispatch(cli.command, &mut context).await
}
