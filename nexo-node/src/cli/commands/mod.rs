//! Command handlers for the `nexo-node` CLI.

pub mod init;
pub mod models;
pub mod start;

pub(crate) use models::ModelsCommand;

use nexo_core::NodeProperties;
use std::path::PathBuf;

pub(crate) fn node_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-node.toml")
}

pub(crate) fn save_node_properties(properties: &NodeProperties) -> nexo_node::Result {
    cli_helpers::config::save(properties, &node_config_path()).map_err(Into::into)
}
