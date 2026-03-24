use crate::shared::types::ModelCategory;
use std::path::PathBuf;
use std::sync::LazyLock;
use tracing::debug;

// ═══════════════════════════════════════════════════════════════════════════
// Manifest types (always available)
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for model component identifiers. Each consumer provides its own enum.
pub trait Component: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Short identifier used as storage key (e.g. "model", "tokenizer").
    fn name(&self) -> &str;

    /// Whether this component is model-specific (stored per-model) or shared
    /// across models of the same family (stored under `shared/<family>/`).
    fn is_model_specific(&self) -> bool;
}

/// A single file to download from HuggingFace.
#[derive(Debug, Clone)]
pub struct ModelFile<C: Component> {
    pub component: C,
    pub hf_repo: String,
    pub hf_filename: String,
    pub size_bytes: u64,
    pub gated: bool,
    /// Expected SHA-256 hex digest. None means not yet collected.
    pub sha256: Option<&'static str>,
}

/// A complete model definition: identity + files to download.
#[derive(Debug, Clone)]
pub struct ModelManifest<C: Component> {
    pub name: String,
    pub family: String,
    pub description: String,
    pub size_gb: f32,
    pub files: Vec<ModelFile<C>>,
}

/// Determine the clean storage path for a model file relative to the models directory.
///
/// - Model-specific components: `<model-name>/<hf_filename>`
/// - Shared components: `shared/<family>/<hf_filename>`
///
/// Model names are sanitized: colons become dashes (e.g. `flux-schnell:q8` -> `flux-schnell-q8`).
pub fn storage_path<C: Component>(manifest: &ModelManifest<C>, file: &ModelFile<C>) -> PathBuf {
    let sanitized_name = manifest.name.replace(':', "-");

    if file.component.is_model_specific() {
        PathBuf::from(&sanitized_name).join(&file.hf_filename)
    } else {
        PathBuf::from("shared")
            .join(&manifest.family)
            .join(&file.hf_filename)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AI-specific component and registry
// ═══════════════════════════════════════════════════════════════════════════

/// Component types for AI model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiComponent {
    Model,
    ModelShard,
    Tokenizer,
    Config,
    Vae,
    TextEncoder,
    ClipEncoder,
    T5Encoder,
}

impl Component for AiComponent {
    fn name(&self) -> &str {
        match self {
            Self::Model => "model",
            Self::ModelShard => "model_shard",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::ClipEncoder => "clip_encoder",
            Self::T5Encoder => "t5_encoder",
        }
    }

    fn is_model_specific(&self) -> bool {
        match self {
            Self::Model | Self::ModelShard | Self::Tokenizer | Self::Config => true,
            Self::Vae | Self::TextEncoder | Self::ClipEncoder | Self::T5Encoder => false,
        }
    }
}

/// An AI model manifest with associated categories.
pub struct AiModelManifest {
    pub manifest: ModelManifest<AiComponent>,
    pub categories: Vec<ModelCategory>,
}

// ── Registry ────────────────────────────────────────────────────────────────

fn build_all_manifests() -> Vec<AiModelManifest> {
    // Start empty -- manifests are added as models are integrated.
    vec![]
}

static ALL_MANIFESTS: LazyLock<Vec<AiModelManifest>> = LazyLock::new(build_all_manifests);

/// Return all known AI model manifests.
pub fn known_manifests() -> &'static [AiModelManifest] {
    &ALL_MANIFESTS
}

/// Look up a manifest by name (case-sensitive).
pub fn find_manifest(name: &str) -> Option<&'static AiModelManifest> {
    ALL_MANIFESTS.iter().find(|m| m.manifest.name == name)
}

/// Return all manifests that belong to a given category.
pub fn manifests_for_category(category: ModelCategory) -> Vec<&'static AiModelManifest> {
    ALL_MANIFESTS
        .iter()
        .filter(|m| m.categories.contains(&category))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Paths
// ═══════════════════════════════════════════════════════════════════════════

/// Default model storage directory: `~/.nexo/local_models/`.
pub fn default_models_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = if let Ok(override_dir) = std::env::var("NEXO_AI_MODELS_DIR") {
            PathBuf::from(override_dir)
        } else {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".nexo")
                .join("local_models")
        };
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

/// Internal hf-hub cache directory: `<models_dir>/.hf-cache/`.
/// Hidden from users; files get hardlinked to clean paths after download.
#[cfg(feature = "download")]
fn hf_cache_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = default_models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// Download (behind #[cfg(feature = "download")])
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "download")]
mod download_impl {
    use super::*;
    use console::Term;
    use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
    use hf_hub::{Cache, Repo, RepoType};
    use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
    use sha2::{Digest, Sha256};
    use thiserror::Error;

    /// Errors that can occur during model download.
    #[derive(Debug, Error)]
    pub enum DownloadError {
        #[error(
            "Model requires access approval on HuggingFace.\n\n  \
            1. Visit: https://huggingface.co/{repo}\n  \
            2. Accept the license agreement\n  \
            3. Create a token at: https://huggingface.co/settings/tokens\n  \
            4. Set: export HF_TOKEN=hf_...\n  \
            5. Retry the pull command"
        )]
        GatedModel { repo: String },

        #[error(
            "Authentication required for repository {repo}.\n\n  \
            1. Create a token at: https://huggingface.co/settings/tokens\n     \
                (select at least \"Read\" access)\n  \
            2. Set: export HF_TOKEN=hf_...\n     \
                Or run: huggingface-cli login\n  \
            3. Retry the pull command"
        )]
        Unauthorized { repo: String },

        #[error("Download failed for {filename} from {repo}: {source}")]
        DownloadFailed {
            repo: String,
            filename: String,
            source: ApiError,
        },

        #[error("Failed to build HuggingFace API client: {0}")]
        ApiSetup(#[from] ApiError),

        #[error("IO error during file placement: {0}")]
        FilePlacement(String),
    }

    // ── HF token resolution ─────────────────────────────────────────────

    fn resolve_hf_token() -> Option<String> {
        if let Ok(token) = std::env::var("HF_TOKEN") {
            let token = token.trim().to_string();
            if !token.is_empty() {
                debug!("Using HuggingFace token from HF_TOKEN environment variable");
                return Some(token);
            }
        }
        let home = dirs::home_dir();
        if let Some(home) = home {
            let token_path = home.join(".nexo").join("hf_token.txt");
            if let Ok(token) = std::fs::read_to_string(&token_path) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    debug!(
                        "Using HuggingFace token from {}",
                        token_path.to_string_lossy()
                    );
                    return Some(token);
                }
            }
        }
        Cache::new(super::hf_cache_dir())
            .token()
            .or_else(|| Cache::from_env().token())
    }

    // ── File placement ──────────────────────────────────────────────────

    fn hardlink_or_copy(src: &std::path::Path, dst: &std::path::Path) -> Result<(), DownloadError> {
        let real_src = src.canonicalize().map_err(|e| {
            DownloadError::FilePlacement(format!(
                "source file not found after download: {} ({e})",
                src.display()
            ))
        })?;

        if dst.exists()
            && let (Ok(src_meta), Ok(dst_meta)) = (real_src.metadata(), dst.metadata())
            && src_meta.len() == dst_meta.len()
        {
            return Ok(());
        }

        if dst.symlink_metadata().is_ok() {
            let _ = std::fs::remove_file(dst);
        }

        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DownloadError::FilePlacement(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        if std::fs::hard_link(&real_src, dst).is_ok() {
            return Ok(());
        }

        std::fs::copy(&real_src, dst).map_err(|e| {
            DownloadError::FilePlacement(format!(
                "failed to copy {} -> {}: {e}",
                real_src.display(),
                dst.display()
            ))
        })?;
        Ok(())
    }

    // ── SHA-256 verification ────────────────────────────────────────────

    fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let digest = format!("{:x}", hasher.finalize());
        Ok(digest == expected)
    }

    // ── Progress bar helpers ────────────────────────────────────────────

    fn truncate_filename(name: &str, max_len: usize) -> String {
        if name.len() <= max_len || max_len < 8 {
            return name.to_string();
        }
        let suffix_len = max_len - 3;
        let start = name.len() - suffix_len;
        format!("...{}", &name[start..])
    }

    fn filename_column_width() -> usize {
        let term_width = Term::stderr().size().1 as usize;
        term_width.saturating_sub(75).max(12)
    }

    #[derive(Clone)]
    struct DownloadProgress {
        bar: ProgressBar,
        max_msg_len: usize,
    }

    impl DownloadProgress {
        fn new(bar: ProgressBar, max_msg_len: usize) -> Self {
            Self { bar, max_msg_len }
        }
    }

    impl Progress for DownloadProgress {
        async fn init(&mut self, size: usize, filename: &str) {
            self.bar.set_length(size as u64);
            self.bar
                .set_message(truncate_filename(filename, self.max_msg_len));
        }

        async fn update(&mut self, size: usize) {
            self.bar.inc(size as u64);
        }

        async fn finish(&mut self) {
            self.bar.finish_with_message("done");
        }
    }

    // ── Core download function ──────────────────────────────────────────

    fn extract_http_status(err: &ApiError) -> Option<u16> {
        if let ApiError::RequestError(reqwest_err) = err {
            reqwest_err.status().map(|s| s.as_u16())
        } else {
            None
        }
    }

    async fn download_file<P: Progress + Clone + Send + Sync + 'static>(
        api: &Api,
        hf_repo: &str,
        hf_filename: &str,
        progress: P,
    ) -> Result<PathBuf, DownloadError> {
        let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));

        match repo.download_with_progress(hf_filename, progress).await {
            Ok(path) => Ok(path),
            Err(e) => {
                let status = extract_http_status(&e);
                let err_str = e.to_string();
                if status == Some(401)
                    || err_str.contains("401")
                    || err_str.contains("Unauthorized")
                {
                    Err(DownloadError::Unauthorized {
                        repo: hf_repo.to_string(),
                    })
                } else if status == Some(403)
                    || err_str.contains("403")
                    || err_str.contains("Forbidden")
                    || err_str.contains("gated")
                    || err_str.contains("Access denied")
                {
                    Err(DownloadError::GatedModel {
                        repo: hf_repo.to_string(),
                    })
                } else {
                    Err(DownloadError::DownloadFailed {
                        repo: hf_repo.to_string(),
                        filename: hf_filename.to_string(),
                        source: e,
                    })
                }
            }
        }
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Download all files for a model manifest with terminal progress bars.
    ///
    /// Returns a list of `(component, clean_path)` pairs. The caller maps
    /// these to their domain-specific model paths struct.
    pub async fn pull_model<C: Component>(
        manifest: &ModelManifest<C>,
    ) -> Result<Vec<(C, PathBuf)>, DownloadError> {
        let mut builder = ApiBuilder::from_env().with_cache_dir(super::hf_cache_dir());
        if let Some(token) = resolve_hf_token() {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build()?;

        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
        let msg_width = filename_column_width();
        let bar_style = ProgressStyle::with_template(&format!(
            "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
        ))
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("━╸─");

        let mdir = default_models_dir();
        let mut downloads: Vec<(C, PathBuf)> = Vec::new();

        for file in &manifest.files {
            let bar = multi.add(ProgressBar::new(file.size_bytes));
            bar.set_style(bar_style.clone());
            bar.set_message(truncate_filename(&file.hf_filename, msg_width));

            let hf_path = download_file(
                &api,
                &file.hf_repo,
                &file.hf_filename,
                DownloadProgress::new(bar, msg_width),
            )
            .await?;

            let clean_rel = storage_path(manifest, file);
            let clean_path = mdir.join(&clean_rel);
            hardlink_or_copy(&hf_path, &clean_path)?;

            if let Some(expected) = file.sha256 {
                match verify_sha256(&clean_path, expected) {
                    Ok(true) => {}
                    Ok(false) => {
                        eprintln!(
                            "warning: SHA-256 mismatch for {} (file may have been updated on HuggingFace)",
                            file.hf_filename
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: failed to verify SHA-256 for {}: {e}",
                            file.hf_filename
                        );
                    }
                }
            }

            downloads.push((file.component.clone(), clean_path));
        }

        Ok(downloads)
    }
}

#[cfg(feature = "download")]
pub use download_impl::{DownloadError, pull_model};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── Test component ──────────────────────────────────────────────────

    #[derive(Clone, Debug)]
    enum TestComponent {
        Model,
        Shared,
    }

    impl Component for TestComponent {
        fn name(&self) -> &str {
            match self {
                Self::Model => "model",
                Self::Shared => "shared",
            }
        }

        fn is_model_specific(&self) -> bool {
            matches!(self, Self::Model)
        }
    }

    fn test_manifest() -> ModelManifest<TestComponent> {
        ModelManifest {
            name: "test-model:q8".to_string(),
            family: "test".to_string(),
            description: "Test model".to_string(),
            size_gb: 1.0,
            files: vec![],
        }
    }

    // ── storage_path ────────────────────────────────────────────────────

    #[test]
    fn model_specific_storage_path() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Model,
            hf_repo: "repo".to_string(),
            hf_filename: "weights.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(path, PathBuf::from("test-model-q8/weights.safetensors"));
    }

    #[test]
    fn shared_storage_path() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Shared,
            hf_repo: "repo".to_string(),
            hf_filename: "vae.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(path, PathBuf::from("shared/test/vae.safetensors"));
    }

    #[test]
    fn storage_path_preserves_subdirectory() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Model,
            hf_repo: "repo".to_string(),
            hf_filename: "subfolder/model.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(
            path,
            PathBuf::from("test-model-q8/subfolder/model.safetensors")
        );
    }

    // ── AiComponent ─────────────────────────────────────────────────────

    #[test]
    fn ai_component_names() {
        assert_eq!(AiComponent::Model.name(), "model");
        assert_eq!(AiComponent::ModelShard.name(), "model_shard");
        assert_eq!(AiComponent::Tokenizer.name(), "tokenizer");
        assert_eq!(AiComponent::Config.name(), "config");
        assert_eq!(AiComponent::Vae.name(), "vae");
        assert_eq!(AiComponent::TextEncoder.name(), "text_encoder");
        assert_eq!(AiComponent::ClipEncoder.name(), "clip_encoder");
        assert_eq!(AiComponent::T5Encoder.name(), "t5_encoder");
    }

    #[test]
    fn ai_component_model_specificity() {
        assert!(AiComponent::Model.is_model_specific());
        assert!(AiComponent::ModelShard.is_model_specific());
        assert!(AiComponent::Tokenizer.is_model_specific());
        assert!(AiComponent::Config.is_model_specific());

        assert!(!AiComponent::Vae.is_model_specific());
        assert!(!AiComponent::TextEncoder.is_model_specific());
        assert!(!AiComponent::ClipEncoder.is_model_specific());
        assert!(!AiComponent::T5Encoder.is_model_specific());
    }

    // ── Registry ────────────────────────────────────────────────────────

    #[test]
    fn known_manifests_starts_empty() {
        // Registry starts empty; manifests are populated as models are integrated.
        assert!(known_manifests().is_empty());
    }

    #[test]
    fn find_manifest_returns_none_for_unknown() {
        assert!(find_manifest("nonexistent-model").is_none());
    }

    #[test]
    fn manifests_for_category_returns_empty() {
        assert!(manifests_for_category(ModelCategory::Chat).is_empty());
    }

    // ── Paths ───────────────────────────────────────────────────────────

    #[test]
    fn default_models_dir_is_under_nexo() {
        let dir = default_models_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains(".nexo") && dir_str.contains("local_models"),
            "expected .nexo/local_models in path, got: {dir_str}"
        );
    }
}
