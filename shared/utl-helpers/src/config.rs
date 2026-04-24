use crate::error::{Error, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::Path;

/// Load a config from the given TOML file path.
/// Returns `C::default()` if the file does not exist.
pub fn load<C: DeserializeOwned + Default>(path: &Path) -> Result<C> {
    if !path.exists() {
        tracing::debug!(
            path = %path.display(),
            "No config file found, using defaults"
        );
        return Ok(C::default());
    }
    load_from_path(path)
}

/// Load config from the given path, creating a default file if missing.
/// Returns the loaded config.
pub fn load_or_create<C: Serialize + DeserializeOwned + Default>(path: &Path) -> Result<C> {
    if !path.exists() {
        let config = C::default();
        save(&config, path)?;
        return Ok(config);
    }

    load_from_path(path)
}

/// Save a config to the given TOML file path (creates parent dirs).
pub fn save<C: Serialize>(config: &C, path: &Path) -> Result {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::Io(format!(
                "Failed to create config directory '{}': {e}",
                parent.display()
            ))
        })?;
    }

    let serialized = toml::to_string_pretty(config)
        .map_err(|e| Error::Config(format!("Failed to serialize config: {e}")))?;

    std::fs::write(path, serialized).map_err(|e| {
        Error::Io(format!(
            "Failed to write config file '{}': {e}",
            path.display()
        ))
    })
}

fn load_from_path<C: DeserializeOwned>(path: &Path) -> Result<C> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        Error::Io(format!(
            "Failed to read config file '{}': {e}",
            path.display()
        ))
    })?;

    let config: C = toml::from_str(&contents).map_err(|e| {
        Error::Config(format!(
            "Failed to parse config file '{}': {e}",
            path.display()
        ))
    })?;

    tracing::info!("Loaded config from {}", path.display());
    Ok(config)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use serde::Deserialize;
    use std::io::Write as _;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
    #[serde(default)]
    struct TestConfig {
        log_level: String,
        custom_field: Option<String>,
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    #[test]
    fn load_returns_defaults_when_no_file() {
        let dir = unique_temp_dir("utl_config_defaults");
        let path = dir.join("nope.toml");
        let config: TestConfig = load(&path).unwrap();
        assert_eq!(config, TestConfig::default());
    }

    #[test]
    fn load_or_create_bootstraps_missing_file() {
        let dir = unique_temp_dir("utl_config_bootstrap");
        let path = dir.join("config.toml");
        let config: TestConfig = load_or_create(&path).unwrap();

        assert!(path.exists());
        assert_eq!(config, TestConfig::default());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn save_and_reload_roundtrip() {
        let dir = unique_temp_dir("utl_config_roundtrip");
        let path = dir.join("config.toml");

        let config = TestConfig {
            log_level: "debug".to_string(),
            custom_field: Some("hello".to_string()),
        };

        save(&config, &path).unwrap();
        let loaded: TestConfig = load(&path).unwrap();
        assert_eq!(config, loaded);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ignores_unknown_sections() {
        let dir = unique_temp_dir("utl_config_unknown");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");

        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"log_level = "info"

[some_future_section]
foo = "bar"
"#
        )
        .unwrap();

        // TestConfig uses serde(default) without deny_unknown_fields
        let config: TestConfig = load(&path).unwrap();
        assert_eq!(config.log_level, "info");

        let _ = std::fs::remove_dir_all(dir);
    }
}
