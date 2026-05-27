use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::api::types::ModelCategory;

use super::coordinator::CoordinatorConfig;
use super::settings::ModelSettings;

/// Top-level configuration for nexo-ai, stored at `~/.nexo/nexo-ai.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    /// Active model name for each category (e.g. `"chat" -> "qwen3-8b"`).
    /// Persisted across sessions - loading/unloading updates this.
    #[serde(alias = "defaults")]
    pub active_models: HashMap<String, String>,
    /// Categories to pre-load on startup.
    pub startup_categories: Vec<String>,
    /// Per-model overrides keyed by model name.
    pub models: HashMap<String, ModelSettings>,
    /// MLX VLM server host (default: `"127.0.0.1"`).
    #[serde(default, alias = "mlx_host")]
    pub mlx_vlm_host: Option<String>,
    /// MLX VLM server port (default: 8080).
    #[serde(default, alias = "mlx_port")]
    pub mlx_vlm_port: Option<u16>,
    /// Path to Python venv containing mlx_vlm (e.g. `"/path/to/.venv"`).
    #[serde(default, alias = "mlx_venv_path")]
    pub mlx_vlm_venv_path: Option<String>,
    /// MLX Audio server host (default: `"127.0.0.1"`).
    #[serde(default)]
    pub mlx_audio_host: Option<String>,
    /// MLX Audio server port (default: 8000).
    #[serde(default)]
    pub mlx_audio_port: Option<u16>,
    /// Path to Python venv containing mlx_audio (e.g. `"/path/to/.venv"`).
    #[serde(default)]
    pub mlx_audio_venv_path: Option<String>,
    /// Hugging Face endpoint used by managed mlx-audio processes.
    #[serde(default)]
    pub mlx_audio_hf_endpoint: Option<String>,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            active_models: HashMap::new(),
            startup_categories: vec!["chat".to_string(), "talk".to_string()],
            models: HashMap::new(),
            mlx_vlm_host: None,
            mlx_vlm_port: None,
            mlx_vlm_venv_path: None,
            mlx_audio_host: None,
            mlx_audio_port: None,
            mlx_audio_venv_path: None,
            mlx_audio_hf_endpoint: None,
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
    pub fn load() -> cli_helpers::Result<Self> {
        cli_helpers::config::load_or_create(&Self::config_path())
    }

    /// Persist the current config to disk.
    pub fn save(&self) -> cli_helpers::Result {
        cli_helpers::config::save(self, &Self::config_path())
    }

    /// Look up the active model name for a given category.
    pub fn active_model_for(&self, category: ModelCategory) -> Option<&str> {
        self.active_models
            .get(category.as_str())
            .map(String::as_str)
    }

    /// Set the active model name for a given category.
    pub fn set_active_model(&mut self, category: ModelCategory, model: String) {
        self.active_models
            .insert(category.as_str().to_string(), model);
    }

    /// Remove the active model for a given category.
    pub fn remove_active_model(&mut self, category: ModelCategory) {
        self.active_models.remove(category.as_str());
    }

    /// Remove all active model assignments.
    pub fn clear_active_models(&mut self) {
        self.active_models.clear();
    }

    /// Return the per-model settings, falling back to defaults if unset.
    pub fn model_settings(&self, name: &str) -> ModelSettings {
        self.models.get(name).cloned().unwrap_or_default()
    }
}

impl From<AiConfig> for CoordinatorConfig {
    fn from(ai: AiConfig) -> Self {
        Self {
            active_models: ai.active_models,
            startup_categories: ai.startup_categories,
            models: ai.models,
            mlx_vlm_host: ai.mlx_vlm_host,
            mlx_vlm_port: ai.mlx_vlm_port,
            mlx_vlm_venv_path: ai.mlx_vlm_venv_path,
            mlx_audio_host: ai.mlx_audio_host,
            mlx_audio_port: ai.mlx_audio_port,
            mlx_audio_venv_path: ai.mlx_audio_venv_path,
            mlx_audio_hf_endpoint: ai.mlx_audio_hf_endpoint,
        }
    }
}

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
    fn default_config_has_empty_active_models() {
        let cfg = AiConfig::default();
        assert!(cfg.active_models.is_empty());
    }

    #[test]
    fn default_config_has_empty_models() {
        let cfg = AiConfig::default();
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn active_model_for_returns_none_when_unset() {
        let cfg = AiConfig::default();
        assert!(cfg.active_model_for(ModelCategory::Chat).is_none());
        assert!(cfg.active_model_for(ModelCategory::Imagine).is_none());
    }

    #[test]
    fn set_active_model_and_retrieve() {
        let mut cfg = AiConfig::default();
        cfg.set_active_model(ModelCategory::Chat, "qwen3-8b".to_string());
        assert_eq!(cfg.active_model_for(ModelCategory::Chat), Some("qwen3-8b"));
    }

    #[test]
    fn set_active_model_overwrites() {
        let mut cfg = AiConfig::default();
        cfg.set_active_model(ModelCategory::Chat, "old-model".to_string());
        cfg.set_active_model(ModelCategory::Chat, "new-model".to_string());
        assert_eq!(cfg.active_model_for(ModelCategory::Chat), Some("new-model"));
    }

    #[test]
    fn remove_active_model() {
        let mut cfg = AiConfig::default();
        cfg.set_active_model(ModelCategory::Chat, "qwen3-8b".to_string());
        assert!(cfg.active_model_for(ModelCategory::Chat).is_some());
        cfg.remove_active_model(ModelCategory::Chat);
        assert!(cfg.active_model_for(ModelCategory::Chat).is_none());
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
        assert_eq!(parsed.active_models.len(), cfg.active_models.len());
    }

    #[test]
    fn serde_roundtrip_with_values() {
        let mut cfg = AiConfig::default();
        cfg.set_active_model(ModelCategory::Chat, "qwen3-8b".to_string());
        cfg.set_active_model(ModelCategory::Imagine, "flux-schnell".to_string());
        cfg.models.insert(
            "qwen3-8b".to_string(),
            ModelSettings {
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(0.9),
                top_k: None,
                seed: Some(42),
                default_steps: None,
                default_guidance: None,
                default_width: None,
                default_height: None,
                voice_description: None,
                max_context_tokens: None,
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

        assert_eq!(
            parsed.active_model_for(ModelCategory::Chat),
            Some("qwen3-8b")
        );
        assert_eq!(
            parsed.active_model_for(ModelCategory::Imagine),
            Some("flux-schnell")
        );
        assert_eq!(parsed.model_settings("qwen3-8b").temperature, Some(0.7));
        assert_eq!(parsed.model_settings("flux-schnell").default_steps, Some(4));
    }

    #[test]
    fn serde_deserialize_empty_table() {
        let toml_str = "";
        let parsed: AiConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            parsed.startup_categories,
            vec!["chat".to_string(), "talk".to_string()]
        );
        assert!(parsed.active_models.is_empty());
    }

    #[test]
    fn config_path_ends_with_expected_segments() {
        let path = AiConfig::config_path();
        let path_str = path.to_string_lossy();
        assert!(path_str.ends_with(".nexo/nexo-ai.toml"));
    }
}
