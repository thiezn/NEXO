use nexo_core::{ClientInfo, DeviceInfo, NodeProperties};
use nexo_node::Result;

use super::{node_config_path, save_node_properties};

/// Create the default node configuration file on disk.
///
/// # Errors
///
/// Returns an error if the default configuration cannot be written.
pub fn run() -> Result {
    let config = NodeProperties::new(
        ClientInfo::new(env!("CARGO_PKG_VERSION")),
        DeviceInfo::default(),
        nexo_ws_schema::AUTH_TOKEN,
    );
    save_node_properties(&config)?;
    let path = node_config_path();
    tracing::info!("Configuration saved to {}", path.display());
    println!("Node configuration initialized at {}", path.display());
    Ok(())
}
