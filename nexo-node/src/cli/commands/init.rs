use crate::config::NodeConfig;

pub fn run() -> utl_helpers::Result {
    let config = NodeConfig::default();
    config.save()?;
    let path = NodeConfig::config_path();
    tracing::info!("Configuration saved to {}", path.display());
    println!("Node configuration initialized at {}", path.display());
    Ok(())
}
