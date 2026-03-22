use crate::manifest::WhisperComponent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Application configuration, stored at `~/.myclaw/speech_to_text.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub default_model: String,
    pub default_language: String,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_model: "whisper-large-v3-turbo".to_string(),
            default_language: "auto".to_string(),
            models: HashMap::new(),
        }
    }
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".myclaw")
            .join("speech_to_text.toml")
    }

    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }

    pub fn model_config(&self, name: &str) -> ModelConfig {
        if let Some(cfg) = self.models.get(name) {
            return cfg.clone();
        }
        let canonical = crate::manifest::resolve_model_name(name);
        if canonical != name {
            if let Some(cfg) = self.models.get(&canonical) {
                return cfg.clone();
            }
        }
        ModelConfig::default()
    }

    pub fn upsert_model(&mut self, name: String, config: ModelConfig) {
        self.models.insert(name, config);
    }
}

/// Per-model configuration (file paths).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelConfig {
    pub model_file: Option<String>,
    pub model_shards: Option<Vec<String>>,
    pub tokenizer: Option<String>,
    pub config_json: Option<String>,
    pub description: Option<String>,
}

/// Resolved model file paths (all PathBuf). Used by inference engine.
#[derive(Debug, Clone)]
pub struct WhisperModelPaths {
    pub model_file: PathBuf,
    pub model_shards: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub config_json: PathBuf,
}

impl WhisperModelPaths {
    pub fn resolve(model_name: &str, config: &AppConfig) -> Option<Self> {
        let model_cfg = config.models.get(model_name).or_else(|| {
            let canonical = crate::manifest::resolve_model_name(model_name);
            config.models.get(&canonical)
        })?;

        let model_file = model_cfg.model_file.as_ref().map(PathBuf::from)?;
        let tokenizer = model_cfg.tokenizer.as_ref().map(PathBuf::from)?;
        let config_json = model_cfg.config_json.as_ref().map(PathBuf::from)?;

        Some(Self {
            model_file,
            model_shards: model_cfg
                .model_shards
                .as_ref()
                .map(|s| s.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            tokenizer,
            config_json,
        })
    }

    /// Build from download results.
    pub fn from_downloads(downloads: &[(WhisperComponent, PathBuf)]) -> Option<Self> {
        let mut model_file = None;
        let mut model_shards = Vec::new();
        let mut tokenizer = None;
        let mut config_json = None;

        for (component, path) in downloads {
            match component {
                WhisperComponent::Model => {
                    if model_file.is_some() {
                        model_shards.push(path.clone());
                    } else {
                        model_file = Some(path.clone());
                    }
                }
                WhisperComponent::Tokenizer => tokenizer = Some(path.clone()),
                WhisperComponent::Config => config_json = Some(path.clone()),
                WhisperComponent::MelFilters => {} // computed, not stored
            }
        }

        Some(Self {
            model_file: model_file?,
            model_shards,
            tokenizer: tokenizer?,
            config_json: config_json?,
        })
    }

    pub fn to_model_config(&self, description: &str) -> ModelConfig {
        ModelConfig {
            model_file: Some(self.model_file.to_string_lossy().to_string()),
            model_shards: if self.model_shards.is_empty() {
                None
            } else {
                Some(
                    self.model_shards
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect(),
                )
            },
            tokenizer: Some(self.tokenizer.to_string_lossy().to_string()),
            config_json: Some(self.config_json.to_string_lossy().to_string()),
            description: Some(description.to_string()),
        }
    }
}

/// Validate that required model files exist on disk.
pub fn validate_paths(paths: &WhisperModelPaths) -> anyhow::Result<()> {
    let check = |path: &Path, name: &str| -> anyhow::Result<()> {
        if !path.exists() {
            anyhow::bail!("{name} not found: {}", path.display());
        }
        Ok(())
    };

    check(&paths.model_file, "Model")?;
    check(&paths.tokenizer, "Tokenizer")?;
    check(&paths.config_json, "Config")?;

    for shard in &paths.model_shards {
        check(shard, "Model shard")?;
    }

    Ok(())
}
