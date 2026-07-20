use nexo_core::{ClientInfo, GatewayProperties};
use std::path::PathBuf;

use crate::memory::db::DbClient;

/// Interactively initialize the gateway configuration and local storage.
///
/// # Errors
///
/// Returns an error when configuration prompts fail, the config cannot be
/// saved, or the database and storage directories cannot be created.
pub async fn run() -> cli_helpers::Result {
    let config_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-gateway.toml");

    tracing::info!("Initializing NEXO Gateway...");

    let host = cli_helpers::interactive::text_input("Bind host", Some("127.0.0.1"))?;

    let port: u16 = cli_helpers::interactive::number_input("Bind port", Some(6969u16))?;

    let config = GatewayProperties::builder(
        ClientInfo::new(env!("CARGO_PKG_VERSION")),
        nexo_ws_schema::AUTH_TOKEN,
    )
    .host(host)
    .port(port)
    .build();

    cli_helpers::config::save(&config, &config_path)?;
    tracing::info!("Config saved to {}", config_path.display());

    // Initialize the database
    let db = DbClient::from_path(config.db_path()).map_err(|error| {
        cli_helpers::Error::Other(format!(
            "Failed to prepare database at '{}': {error}",
            config.db_path().display()
        ))
    })?;
    db.initialize_schema().await.map_err(|error| {
        cli_helpers::Error::Other(format!(
            "Failed to initialize database schema at '{}': {error}",
            config.db_path().display()
        ))
    })?;

    // Pre-create markdown storage directory
    let storage_root = cli_helpers::resolve_path_str(config.storage_root())?;
    let markdown_dir = storage_root.join("markdown");
    std::fs::create_dir_all(&markdown_dir).map_err(|e| {
        cli_helpers::Error::Other(format!(
            "Failed to create markdown dir '{}': {e}",
            markdown_dir.display()
        ))
    })?;

    println!("NEXO Gateway initialized successfully.");
    println!("  Config: {}", config_path.display());
    println!("  Database: {}", config.db_path().display());
    println!("  Markdown storage: {}", markdown_dir.display());
    println!("\nRun `nexo start` to start the gateway.");

    Ok(())
}
