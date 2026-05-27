#[cfg(feature = "cli")]
#[tokio::main]
async fn main() {
    use clap::Parser;
    let cli = nexo_ai::cli::base::Cli::parse();
    cli_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

    if let Err(e) = nexo_ai::cli::commands::dispatch(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!(
        "nexo-ai binary requires the 'cli' feature. Build with: cargo build -p nexo-ai --features cli"
    );
    std::process::exit(1);
}
