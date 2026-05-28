//! Binary entry point for the NEXO gateway executable.

mod agent;
mod cli;
mod config;
mod init;
mod memory;
mod runtime;
mod schema_cmd;
mod server;
mod tools;

use clap::Parser;
use cli::{Cli, Command};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = cli_helpers::init_tracing(cli.log_level, cli.no_color) {
        eprintln!("Failed to initialize tracing: {error}");
    }

    if let Err(e) = run(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

async fn run(command: Command) -> cli_helpers::Result {
    match command {
        Command::Init => init::run_init().await,
        Command::Start { host, port } => {
            let mut config = config::GatewayConfig::load()?;
            if let Some(h) = host {
                config.host = h;
            }
            if let Some(p) = port {
                config.port = p;
            }
            runtime::run(&config).await
        }
        Command::Schema { section, output } => schema_cmd::run_schema(section, output.as_deref()),
    }
}
