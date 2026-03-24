use crate::config::GatewayConfig;

pub async fn run_init() -> utl_helpers::Result {
    tracing::info!("Initializing NEXO Gateway...");

    let host = utl_helpers::interactive::text_input("Bind host", Some("127.0.0.1"))?;

    let port: u16 = utl_helpers::interactive::number_input("Bind port", Some(6969u16))?;

    let log_levels = &["trace", "debug", "info", "warn", "error"];
    let log_level_idx =
        utl_helpers::interactive::select("Default log level", log_levels, Some(2))?;
    let log_level = log_levels[log_level_idx];

    let tick_interval_ms: u64 =
        utl_helpers::interactive::number_input("Tick interval (ms)", Some(15000u64))?;

    let config = GatewayConfig {
        host,
        port,
        log_level: log_level.to_string(),
        tick_interval_ms,
        ..Default::default()
    };

    config.save()?;
    tracing::info!(
        "Config saved to {}",
        GatewayConfig::config_path().display()
    );

    // Initialize the database
    let db_path = utl_helpers::resolve_path_str(&config.db_path)?;
    crate::memory::persistent::initialize(&db_path).await?;

    println!("NEXO Gateway initialized successfully.");
    println!("  Config: {}", GatewayConfig::config_path().display());
    println!("  Database: {}", db_path.display());
    println!("\nRun `nexo start` to start the gateway.");

    Ok(())
}
