use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::shared::types::ModelCategory;

// ── AiConfig ────────────────────────────────────────────────────────────────

/// Top-level configuration for nexo-ai, stored at `~/.nexo/nexo-ai.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    /// Default model name for each category (e.g. `"chat" -> "qwen3-8b"`).
    pub defaults: HashMap<String, String>,
    /// Categories to pre-load on startup.
    pub startup_categories: Vec<String>,
    /// Per-model overrides keyed by model name.
    pub models: HashMap<String, ModelSettings>,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            defaults: HashMap::new(),
            startup_categories: vec!["chat".to_string(), "talk".to_string()],
            models: HashMap::new(),
        }
    }
}

impl AiConfig {
    /// Canonical path to the config file.
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("nexo-ai.toml")
    }

    /// Load config from disk, creating a default file if it does not exist.
    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    /// Persist the current config to disk.
    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }

    /// Look up the default model name for a given category.
    pub fn default_for(&self, category: ModelCategory) -> Option<&str> {
        self.defaults.get(category.as_str()).map(String::as_str)
    }

    /// Set the default model name for a given category.
    pub fn set_default(&mut self, category: ModelCategory, model: String) {
        self.defaults.insert(category.as_str().to_string(), model);
    }

    /// Return the per-model settings, falling back to defaults if unset.
    pub fn model_settings(&self, name: &str) -> ModelSettings {
        self.models.get(name).cloned().unwrap_or_default()
    }
}

// ── ModelSettings ───────────────────────────────────────────────────────────

/// Per-model runtime overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelSettings {
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub default_steps: Option<u32>,
    pub default_guidance: Option<f64>,
    pub default_width: Option<u32>,
    pub default_height: Option<u32>,
    pub voice_description: Option<String>,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_chat_startup() {
        let cfg = AiConfig::default();
        assert_eq!(
            cfg.startup_categories,
            vec!["chat".to_string(), "talk".to_string()]
        );
    }

    #[test]
    fn default_config_has_empty_defaults() {
        let cfg = AiConfig::default();
        assert!(cfg.defaults.is_empty());
    }

    #[test]
    fn default_config_has_empty_models() {
        let cfg = AiConfig::default();
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn default_for_returns_none_when_unset() {
        let cfg = AiConfig::default();
        assert!(cfg.default_for(ModelCategory::Chat).is_none());
        assert!(cfg.default_for(ModelCategory::Imagine).is_none());
    }

    #[test]
    fn set_default_and_retrieve() {
        let mut cfg = AiConfig::default();
        cfg.set_default(ModelCategory::Chat, "qwen3-8b".to_string());
        assert_eq!(cfg.default_for(ModelCategory::Chat), Some("qwen3-8b"));
    }

    #[test]
    fn set_default_overwrites() {
        let mut cfg = AiConfig::default();
        cfg.set_default(ModelCategory::Chat, "old-model".to_string());
        cfg.set_default(ModelCategory::Chat, "new-model".to_string());
        assert_eq!(cfg.default_for(ModelCategory::Chat), Some("new-model"));
    }

    #[test]
    fn model_settings_returns_default_when_unset() {
        let cfg = AiConfig::default();
        let settings = cfg.model_settings("nonexistent");
        assert!(settings.temperature.is_none());
        assert!(settings.max_tokens.is_none());
        assert!(settings.seed.is_none());
    }

    #[test]
    fn model_settings_returns_stored_values() {
        let mut cfg = AiConfig::default();
        cfg.models.insert(
            "my-model".to_string(),
            ModelSettings {
                temperature: Some(0.8),
                max_tokens: Some(2048),
                ..Default::default()
            },
        );
        let settings = cfg.model_settings("my-model");
        assert_eq!(settings.temperature, Some(0.8));
        assert_eq!(settings.max_tokens, Some(2048));
    }

    #[test]
    fn serde_roundtrip_default() {
        let cfg = AiConfig::default();
        let toml_str = toml::to_string(&cfg).unwrap();
        let parsed: AiConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.startup_categories, cfg.startup_categories);
        assert_eq!(parsed.defaults.len(), cfg.defaults.len());
    }

    #[test]
    fn serde_roundtrip_with_values() {
        let mut cfg = AiConfig::default();
        cfg.set_default(ModelCategory::Chat, "qwen3-8b".to_string());
        cfg.set_default(ModelCategory::Imagine, "flux-schnell".to_string());
        cfg.models.insert(
            "qwen3-8b".to_string(),
            ModelSettings {
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(0.9),
                seed: Some(42),
                default_steps: None,
                default_guidance: None,
                default_width: None,
                default_height: None,
                voice_description: None,
            },
        );
        cfg.models.insert(
            "flux-schnell".to_string(),
            ModelSettings {
                default_steps: Some(4),
                default_guidance: Some(0.0),
                default_width: Some(1024),
                default_height: Some(1024),
                ..Default::default()
            },
        );

        let toml_str = toml::to_string(&cfg).unwrap();
        let parsed: AiConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.default_for(ModelCategory::Chat), Some("qwen3-8b"));
        assert_eq!(
            parsed.default_for(ModelCategory::Imagine),
            Some("flux-schnell")
        );
        assert_eq!(parsed.model_settings("qwen3-8b").temperature, Some(0.7));
        assert_eq!(parsed.model_settings("flux-schnell").default_steps, Some(4));
    }

    #[test]
    fn serde_deserialize_empty_table() {
        let toml_str = "";
        let parsed: AiConfig = toml::from_str(toml_str).unwrap();
        // Should fall back to defaults
        assert_eq!(parsed.startup_categories, vec!["chat".to_string(), "talk".to_string()]);
        assert!(parsed.defaults.is_empty());
    }

    #[test]
    fn model_settings_serde_roundtrip() {
        let settings = ModelSettings {
            temperature: Some(0.5),
            max_tokens: Some(1024),
            top_p: Some(0.95),
            seed: Some(123),
            default_steps: Some(20),
            default_guidance: Some(7.5),
            default_width: Some(512),
            default_height: Some(512),
            voice_description: Some("warm male voice".to_string()),
        };
        let toml_str = toml::to_string(&settings).unwrap();
        let parsed: ModelSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.temperature, Some(0.5));
        assert_eq!(parsed.max_tokens, Some(1024));
        assert_eq!(parsed.seed, Some(123));
        assert_eq!(parsed.default_steps, Some(20));
        assert_eq!(parsed.default_guidance, Some(7.5));
        assert_eq!(
            parsed.voice_description,
            Some("warm male voice".to_string())
        );
    }

    #[test]
    fn config_path_ends_with_expected_segments() {
        let path = AiConfig::config_path();
        let path_str = path.to_string_lossy();
        assert!(path_str.ends_with(".nexo/nexo-ai.toml"));
    }
}
