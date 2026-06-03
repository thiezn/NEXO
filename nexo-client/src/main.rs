mod audio;
mod cli;
mod config;
mod tui;
mod vision;

use clap::Parser;
use cli::base::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = cli_helpers::init_tracing(cli.log_level, cli.no_color) {
        eprintln!("Failed to initialize tracing: {error}");
    }

    if let Err(e) = cli::commands::dispatch(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
