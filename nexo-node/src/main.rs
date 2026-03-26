mod cli;
mod config;
mod connect;
mod registry;

use clap::Parser;
use cli::{Cli, Command};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

    if let Err(e) = run(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

async fn run(command: Command) -> utl_helpers::Result {
    match command {
        Command::Init => {
            let config = config::NodeConfig::default();
            config.save()?;
            let path = config::NodeConfig::config_path();
            tracing::info!("Configuration saved to {}", path.display());
            println!("Node configuration initialized at {}", path.display());
            Ok(())
        }
        Command::Start { url } => {
            let mut config = config::NodeConfig::load()?;
            if let Some(u) = url {
                config.gateway_url = u;
            }

            tracing::info!(
                "Starting nexo-node '{}' v{}",
                config.node_id,
                config.node_version
            );

            let registry = registry::ToolRegistry::with_builtins();
            tracing::info!(
                "Loaded {} tool(s): {:?}",
                registry.tool_count(),
                registry.specs().iter().map(|s| &s.name).collect::<Vec<_>>()
            );

            connect::run_node(&config, &registry).await
        }
    }
}
