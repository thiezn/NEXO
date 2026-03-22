use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Application configuration, stored at `~/.myclaw/text_to_speech.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub default_model: String,
    pub default_description: String,
    pub default_max_tokens: usize,
    pub default_temperature: f64,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_model: "parler-mini".to_string(),
            default_description: "A clear, natural speaking voice at a moderate pace.".to_string(),
            default_max_tokens: 2580,
            default_temperature: 1.0,
            models: HashMap::new(),
        }
    }
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".myclaw")
            .join("text_to_speech.toml")
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
        if canonical != name
            && let Some(cfg) = self.models.get(&canonical)
        {
            return cfg.clone();
        }
        ModelConfig::default()
    }

    pub fn upsert_model(&mut self, name: String, config: ModelConfig) {
        self.models.insert(name, config);
    }
}

/// Per-model configuration (paths + synthesis defaults).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelConfig {
    pub decoder: Option<String>,
    pub decoder_shards: Option<Vec<String>>,
    pub tokenizer: Option<String>,
    pub config_json: Option<String>,

    pub default_max_tokens: Option<usize>,
    pub default_temperature: Option<f64>,
    pub description: Option<String>,
    pub family: Option<String>,
    pub sample_rate: Option<u32>,
}

impl ModelConfig {
    pub fn effective_max_tokens(&self, global: &AppConfig) -> usize {
        self.default_max_tokens.unwrap_or(global.default_max_tokens)
    }

    pub fn effective_temperature(&self, global: &AppConfig) -> f64 {
        self.default_temperature.unwrap_or(global.default_temperature)
    }
}

/// Resolved model file paths. Used by inference engines.
#[derive(Debug, Clone)]
pub struct TTSModelPaths {
    pub model_file: PathBuf,
    pub model_shards: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub config_json: PathBuf,
}

impl TTSModelPaths {
    pub fn resolve(model_name: &str, config: &AppConfig) -> Option<Self> {
        let model_cfg = config.models.get(model_name).or_else(|| {
            let canonical = crate::manifest::resolve_model_name(model_name);
            config.models.get(&canonical)
        })?;

        let decoder = model_cfg.decoder.as_ref().map(PathBuf::from)?;
        let tokenizer = model_cfg.tokenizer.as_ref().map(PathBuf::from)?;
        let config_json = model_cfg.config_json.as_ref().map(PathBuf::from)?;

        Some(Self {
            model_file: decoder,
            model_shards: model_cfg
                .decoder_shards
                .as_ref()
                .map(|s| s.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            tokenizer,
            config_json,
        })
    }

    pub fn from_downloads(
        downloads: &[(crate::manifest::TTSComponent, PathBuf)],
    ) -> Option<Self> {
        use crate::manifest::TTSComponent;

        let mut decoder = None;
        let mut decoder_shards = Vec::new();
        let mut tokenizer = None;
        let mut config_json = None;

        for (component, path) in downloads {
            match component {
                TTSComponent::Decoder => decoder = Some(path.clone()),
                TTSComponent::DecoderShard => decoder_shards.push(path.clone()),
                TTSComponent::Tokenizer => tokenizer = Some(path.clone()),
                TTSComponent::Config => config_json = Some(path.clone()),
            }
        }

        // Use first shard as model_file if no single decoder file
        let model_file = decoder.or_else(|| decoder_shards.first().cloned())?;
        let tokenizer = tokenizer?;
        let config_json = config_json?;

        Some(Self {
            model_file,
            model_shards: decoder_shards,
            tokenizer,
            config_json,
        })
    }

    pub fn to_model_config(&self, family: &str, description: &str) -> ModelConfig {
        ModelConfig {
            decoder: Some(self.model_file.to_string_lossy().to_string()),
            decoder_shards: if self.model_shards.is_empty() {
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
            family: Some(family.to_string()),
            ..Default::default()
        }
    }

    /// All safetensor files for this model (single file or shards).
    pub fn safetensor_files(&self) -> Vec<&Path> {
        if self.model_shards.is_empty() {
            vec![&self.model_file]
        } else {
            self.model_shards.iter().map(|p| p.as_path()).collect()
        }
    }
}

/// Validate that required model files exist on disk.
pub fn validate_paths(paths: &TTSModelPaths) -> anyhow::Result<()> {
    let check = |path: &Path, name: &str| -> anyhow::Result<()> {
        if !path.exists() {
            anyhow::bail!("{name} not found: {}", path.display());
        }
        Ok(())
    };

    if paths.model_shards.is_empty() {
        check(&paths.model_file, "Model")?;
    } else {
        for shard in &paths.model_shards {
            check(shard, "Model shard")?;
        }
    }
    check(&paths.tokenizer, "Tokenizer")?;
    check(&paths.config_json, "Config")?;

    Ok(())
}
