mod agent;
mod cli;
mod config;
mod init;
mod memory;
mod schema_cmd;
mod server;

use clap::Parser;
use cli::{Cli, Command};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    cli_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

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
            server::run(&config).await
        }
        Command::Schema { section, output } => schema_cmd::run_schema(section, output.as_deref()),
    }
}
