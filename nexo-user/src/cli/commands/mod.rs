pub mod schema;
pub mod start;

use nexo_core::UserProperties;
use std::path::PathBuf;

pub(crate) fn user_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-user.toml")
}

pub(crate) fn save_user_properties(properties: &UserProperties) -> cli_helpers::Result {
    cli_helpers::config::save(properties, &user_config_path())
}
