mod cli;
mod config;
mod tui;

use clap::Parser;
use cli::base::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    if let Err(e) = cli::commands::dispatch(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
