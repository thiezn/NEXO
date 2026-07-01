//! Command-line entry point for the `nexo-user` binary.

// mod cli;

// use clap::Parser;
// use cli::base::Cli;
// use std::process::ExitCode;

// #[tokio::main]
// async fn main() -> Result<ExitCode> {
//     let cli = Cli::parse();
//     cli.common.init_tracing()?;

//     let mut context = cli.common.command_context()?;
//     cli::base::dispatch(cli.command, &mut context).await
// }

mod cli;

use std::error::Error as StdError;
use std::process::ExitCode;

use clap::Parser;
use cli::base::Cli;

#[tokio::main]
async fn main() -> Result<ExitCode, Box<dyn StdError>> {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    let mut context = cli.common.command_context()?;
    let exit_code = cli::base::dispatch(cli.command, &mut context).await?;
    Ok(exit_code)
}
