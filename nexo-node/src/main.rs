//! Command-line entry point for the `nexo-node` binary.

mod cli;
mod config;
mod inference;
mod tools;
mod transport;

use clap::Parser;
use cli::base::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    cli_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

    if let Err(e) = cli::base::dispatch(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
