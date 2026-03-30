mod chat;
mod cli;
mod config;
mod connect;
mod schema_cmd;

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
        Command::Connect { url } => connect::run_connect(url).await,
        Command::Chat {
            url,
            session,
            name,
            model,
        } => {
            chat::run_chat(chat::ChatOptions {
                url_override: url,
                session_id: session,
                session_name: name,
                model_id: model,
            })
            .await
        }
        Command::Schema { section, output } => schema_cmd::run_schema(section, output.as_deref()),
    }
}
