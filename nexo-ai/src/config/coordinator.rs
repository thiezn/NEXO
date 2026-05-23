use std::collections::HashMap;

use crate::api::types::ModelCategory;

use super::settings::ModelSettings;

/// Lightweight config consumed by the coordinator. No file I/O.
/// Both `AiConfig` and external crates (e.g. `nexo-node`) can produce this.
#[derive(Debug, Clone, Default)]
pub struct CoordinatorConfig {
    /// Active model name for each category (e.g. `"chat" -> "gemma-4-e4b-it"`).
    pub active_models: HashMap<String, String>,
    /// Categories to pre-load on startup.
    pub startup_categories: Vec<String>,
    /// Per-model overrides keyed by model name.
    pub models: HashMap<String, ModelSettings>,
    /// MLX VLM server host (default: `"127.0.0.1"`).
    pub mlx_vlm_host: Option<String>,
    /// MLX VLM server port (default: 8080).
    pub mlx_vlm_port: Option<u16>,
    /// Path to Python venv containing mlx_vlm (e.g. `"/path/to/.venv"`).
    pub mlx_vlm_venv_path: Option<String>,
    /// MLX Audio server host (default: `"127.0.0.1"`).
    pub mlx_audio_host: Option<String>,
    /// MLX Audio server port (default: 8000).
    pub mlx_audio_port: Option<u16>,
    /// Path to Python venv containing mlx_audio (e.g. `"/path/to/.venv"`).
    pub mlx_audio_venv_path: Option<String>,
    /// Hugging Face endpoint used by managed mlx-audio processes.
    pub mlx_audio_hf_endpoint: Option<String>,
}

impl CoordinatorConfig {
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

    /// MLX VLM server address (host, port) with defaults.
    pub fn mlx_vlm_server_addr(&self) -> (String, u16) {
        (
            self.mlx_vlm_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            self.mlx_vlm_port.unwrap_or(8080),
        )
    }

    /// MLX Audio server address (host, port) with defaults.
    pub fn mlx_audio_server_addr(&self) -> (String, u16) {
        (
            self.mlx_audio_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            self.mlx_audio_port.unwrap_or(8000),
        )
    }

    /// Hugging Face endpoint used by managed mlx-audio processes.
    pub fn mlx_audio_hf_endpoint(&self) -> String {
        self.mlx_audio_hf_endpoint
            .clone()
            .unwrap_or_else(|| "https://hf-mirror.com".to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::config::AiConfig;

    #[test]
    fn coordinator_config_from_ai_config() {
        let mut ai = AiConfig::default();
        ai.set_active_model(ModelCategory::Chat, "test-model".to_string());
        ai.models.insert(
            "test-model".to_string(),
            ModelSettings {
                temperature: Some(0.8),
                ..Default::default()
            },
        );

        let coord: CoordinatorConfig = ai.into();
        assert_eq!(
            coord.active_model_for(ModelCategory::Chat),
            Some("test-model")
        );
        assert_eq!(coord.model_settings("test-model").temperature, Some(0.8));
        assert_eq!(
            coord.startup_categories,
            vec!["chat".to_string(), "talk".to_string()]
        );
    }

    #[test]
    fn coordinator_config_active_model_lifecycle() {
        let mut config = CoordinatorConfig::default();
        assert!(config.active_model_for(ModelCategory::Chat).is_none());

        config.set_active_model(ModelCategory::Chat, "m1".to_string());
        assert_eq!(config.active_model_for(ModelCategory::Chat), Some("m1"));

        config.remove_active_model(ModelCategory::Chat);
        assert!(config.active_model_for(ModelCategory::Chat).is_none());
    }

    #[test]
    fn coordinator_config_clear_active_models() {
        let mut config = CoordinatorConfig::default();
        config.set_active_model(ModelCategory::Chat, "m1".to_string());
        config.set_active_model(ModelCategory::Tool, "m2".to_string());
        config.clear_active_models();
        assert!(config.active_models.is_empty());
    }

    #[test]
    fn coordinator_config_model_settings_fallback() {
        let config = CoordinatorConfig::default();
        let settings = config.model_settings("nonexistent");
        assert!(settings.temperature.is_none());
    }
}
