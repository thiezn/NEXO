use nexo_node::{NexoNodeConfig, Result};

/// Create the default node configuration file on disk.
///
/// # Errors
///
/// Returns an error if the default configuration cannot be written.
pub fn run() -> Result {
    let config = NexoNodeConfig::default();
    config.save()?;
    let path = NexoNodeConfig::config_path();
    tracing::info!("Configuration saved to {}", path.display());
    println!("Node configuration initialized at {}", path.display());
    Ok(())
}
