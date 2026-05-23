use crate::config::GatewayConfig;

/// Interactively initialize the gateway configuration and local storage.
///
/// # Errors
///
/// Returns an error when configuration prompts fail, the config cannot be
/// saved, or the database and storage directories cannot be created.
pub async fn run_init() -> cli_helpers::Result {
    tracing::info!("Initializing NEXO Gateway...");

    let host = cli_helpers::interactive::text_input("Bind host", Some("127.0.0.1"))?;

    let port: u16 = cli_helpers::interactive::number_input("Bind port", Some(6969u16))?;

    let log_levels = &["trace", "debug", "info", "warn", "error"];
    let log_level_idx = cli_helpers::interactive::select("Default log level", log_levels, Some(2))?;
    let log_level = log_levels[log_level_idx];

    let tick_interval_ms: u64 =
        cli_helpers::interactive::number_input("Tick interval (ms)", Some(15000u64))?;

    let config = GatewayConfig {
        host,
        port,
        log_level: log_level.to_string(),
        tick_interval_ms,
        ..Default::default()
    };

    config.save()?;
    tracing::info!("Config saved to {}", GatewayConfig::config_path().display());

    // Initialize the database
    let db_path = cli_helpers::resolve_path_str(&config.db_path)?;
    crate::memory::persistent::initialize(&db_path).await?;

    // Pre-create markdown storage directory
    let storage_root = cli_helpers::resolve_path_str(&config.storage_root)?;
    let markdown_dir = storage_root.join("markdown");
    std::fs::create_dir_all(&markdown_dir).map_err(|e| {
        cli_helpers::Error::Io(format!(
            "Failed to create markdown dir '{}': {e}",
            markdown_dir.display()
        ))
    })?;

    println!("NEXO Gateway initialized successfully.");
    println!("  Config: {}", GatewayConfig::config_path().display());
    println!("  Database: {}", db_path.display());
    println!("  Markdown storage: {}", markdown_dir.display());
    println!("\nRun `nexo start` to start the gateway.");

    Ok(())
}
