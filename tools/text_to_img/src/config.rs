use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Application configuration, stored at `~/.myclaw/text_to_img.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub default_model: String,
    pub default_width: u32,
    pub default_height: u32,
    pub default_steps: u32,
    pub t5_variant: Option<String>,
    pub qwen3_variant: Option<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_model: "flux-schnell:q8".to_string(),
            default_width: 1024,
            default_height: 1024,
            default_steps: 4,
            t5_variant: None,
            qwen3_variant: None,
            models: HashMap::new(),
        }
    }
}

impl AppConfig {
    /// Config file path: `~/.myclaw/text_to_img.toml`.
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".myclaw")
            .join("text_to_img.toml")
    }

    /// Load config from disk, creating a default if missing.
    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    /// Save config to disk.
    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }

    /// Get model config by name, falling back to an empty default.
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

    /// Insert or update a model config entry.
    pub fn upsert_model(&mut self, name: String, config: ModelConfig) {
        self.models.insert(name, config);
    }
}

/// Per-model configuration (paths + generation defaults).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelConfig {
    pub transformer: Option<String>,
    pub transformer_shards: Option<Vec<String>>,
    pub vae: Option<String>,
    pub t5_encoder: Option<String>,
    pub clip_encoder: Option<String>,
    pub t5_tokenizer: Option<String>,
    pub clip_tokenizer: Option<String>,
    pub text_encoder_files: Option<Vec<String>>,
    pub text_tokenizer: Option<String>,

    pub default_steps: Option<u32>,
    pub default_guidance: Option<f64>,
    pub default_width: Option<u32>,
    pub default_height: Option<u32>,
    pub is_schnell: Option<bool>,
    pub description: Option<String>,
    pub family: Option<String>,
}

impl ModelConfig {
    pub fn effective_steps(&self, global: &AppConfig) -> u32 {
        self.default_steps.unwrap_or(global.default_steps)
    }

    pub fn effective_guidance(&self) -> f64 {
        self.default_guidance.unwrap_or(3.5)
    }

    pub fn effective_width(&self, global: &AppConfig) -> u32 {
        self.default_width.unwrap_or(global.default_width)
    }

    pub fn effective_height(&self, global: &AppConfig) -> u32 {
        self.default_height.unwrap_or(global.default_height)
    }
}

/// Resolved model file paths (all PathBuf). Used by inference engines.
#[derive(Debug, Clone)]
pub struct ImageModelPaths {
    pub transformer: PathBuf,
    pub transformer_shards: Vec<PathBuf>,
    pub vae: PathBuf,
    pub t5_encoder: Option<PathBuf>,
    pub clip_encoder: Option<PathBuf>,
    pub t5_tokenizer: Option<PathBuf>,
    pub clip_tokenizer: Option<PathBuf>,
    pub text_encoder_files: Vec<PathBuf>,
    pub text_tokenizer: Option<PathBuf>,
}

impl ImageModelPaths {
    /// Resolve paths from config. Returns None if transformer/VAE not configured.
    pub fn resolve(model_name: &str, config: &AppConfig) -> Option<Self> {
        let model_cfg = config.models.get(model_name).or_else(|| {
            let canonical = crate::manifest::resolve_model_name(model_name);
            config.models.get(&canonical)
        })?;

        let transformer = model_cfg.transformer.as_ref().map(PathBuf::from)?;
        let vae = model_cfg.vae.as_ref().map(PathBuf::from)?;

        Some(Self {
            transformer,
            transformer_shards: model_cfg
                .transformer_shards
                .as_ref()
                .map(|s| s.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            vae,
            t5_encoder: model_cfg.t5_encoder.as_ref().map(PathBuf::from),
            clip_encoder: model_cfg.clip_encoder.as_ref().map(PathBuf::from),
            t5_tokenizer: model_cfg.t5_tokenizer.as_ref().map(PathBuf::from),
            clip_tokenizer: model_cfg.clip_tokenizer.as_ref().map(PathBuf::from),
            text_encoder_files: model_cfg
                .text_encoder_files
                .as_ref()
                .map(|f| f.iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            text_tokenizer: model_cfg.text_tokenizer.as_ref().map(PathBuf::from),
        })
    }

    /// Build from download results. Maps component names to path fields.
    pub fn from_downloads(downloads: &[(crate::manifest::ImageComponent, PathBuf)]) -> Option<Self> {
        use crate::manifest::ImageComponent;

        let mut transformer = None;
        let mut transformer_shards = Vec::new();
        let mut vae = None;
        let mut t5_encoder = None;
        let mut clip_encoder = None;
        let mut t5_tokenizer = None;
        let mut clip_tokenizer = None;
        let mut text_encoder_files = Vec::new();
        let mut text_tokenizer = None;

        for (component, path) in downloads {
            match component {
                ImageComponent::Transformer => transformer = Some(path.clone()),
                ImageComponent::TransformerShard => transformer_shards.push(path.clone()),
                ImageComponent::Vae => vae = Some(path.clone()),
                ImageComponent::T5Encoder => t5_encoder = Some(path.clone()),
                ImageComponent::ClipEncoder => clip_encoder = Some(path.clone()),
                ImageComponent::T5Tokenizer => t5_tokenizer = Some(path.clone()),
                ImageComponent::ClipTokenizer => clip_tokenizer = Some(path.clone()),
                ImageComponent::TextEncoder => text_encoder_files.push(path.clone()),
                ImageComponent::TextTokenizer => text_tokenizer = Some(path.clone()),
            }
        }

        // Use first shard as transformer if no single transformer file
        let transformer = transformer.or_else(|| transformer_shards.first().cloned())?;
        let vae = vae?;

        Some(Self {
            transformer,
            transformer_shards,
            vae,
            t5_encoder,
            clip_encoder,
            t5_tokenizer,
            clip_tokenizer,
            text_encoder_files,
            text_tokenizer,
        })
    }

    /// Convert to a ModelConfig for saving to TOML.
    pub fn to_model_config(&self, family: &str, description: &str, defaults: &local_inference_helpers::manifest::ManifestDefaults, is_schnell: bool) -> ModelConfig {
        ModelConfig {
            transformer: Some(self.transformer.to_string_lossy().to_string()),
            transformer_shards: if self.transformer_shards.is_empty() {
                None
            } else {
                Some(self.transformer_shards.iter().map(|p| p.to_string_lossy().to_string()).collect())
            },
            vae: Some(self.vae.to_string_lossy().to_string()),
            t5_encoder: self.t5_encoder.as_ref().map(|p| p.to_string_lossy().to_string()),
            clip_encoder: self.clip_encoder.as_ref().map(|p| p.to_string_lossy().to_string()),
            t5_tokenizer: self.t5_tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
            clip_tokenizer: self.clip_tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
            text_encoder_files: if self.text_encoder_files.is_empty() {
                None
            } else {
                Some(self.text_encoder_files.iter().map(|p| p.to_string_lossy().to_string()).collect())
            },
            text_tokenizer: self.text_tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
            default_steps: Some(defaults.steps),
            default_guidance: Some(defaults.guidance),
            default_width: Some(defaults.width),
            default_height: Some(defaults.height),
            is_schnell: Some(is_schnell),
            description: Some(description.to_string()),
            family: Some(family.to_string()),
        }
    }
}

/// Validate that required model files exist on disk.
pub fn validate_paths(paths: &ImageModelPaths) -> anyhow::Result<()> {
    let check = |path: &Path, name: &str| -> anyhow::Result<()> {
        if !path.exists() {
            anyhow::bail!("{name} not found: {}", path.display());
        }
        Ok(())
    };

    check(&paths.transformer, "Transformer")?;
    check(&paths.vae, "VAE")?;

    for shard in &paths.transformer_shards {
        check(shard, "Transformer shard")?;
    }
    if let Some(ref p) = paths.t5_encoder {
        check(p, "T5 encoder")?;
    }
    if let Some(ref p) = paths.clip_encoder {
        check(p, "CLIP encoder")?;
    }
    if let Some(ref p) = paths.t5_tokenizer {
        check(p, "T5 tokenizer")?;
    }
    if let Some(ref p) = paths.clip_tokenizer {
        check(p, "CLIP tokenizer")?;
    }
    for f in &paths.text_encoder_files {
        check(f, "Text encoder")?;
    }
    if let Some(ref p) = paths.text_tokenizer {
        check(p, "Text tokenizer")?;
    }

    Ok(())
}
