mod cli;
mod config;
mod connect;
mod download;
mod kv_cache;
mod registry;

use clap::Parser;
use cli::base::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

    if let Err(e) = cli::commands::dispatch(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
