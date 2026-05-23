//! `init` command implementation.

use crate::config::NodeConfig;

/// Create the default node configuration file on disk.
///
/// # Errors
///
/// Returns an error if the default configuration cannot be written.
pub fn run() -> cli_helpers::Result {
    let config = NodeConfig::default();
    config.save()?;
    let path = NodeConfig::config_path();
    tracing::info!("Configuration saved to {}", path.display());
    println!("Node configuration initialized at {}", path.display());
    Ok(())
}
