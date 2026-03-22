use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use thiserror::Error;

use crate::manifest::{Component, ModelManifest, storage_path};
use crate::paths::{default_models_dir, hf_cache_dir};

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

    #[error("Failed to build sync HuggingFace API client: {0}")]
    SyncApiSetup(String),

    #[error("Sync download failed for {filename} from {repo}: {message}")]
    SyncDownloadFailed {
        repo: String,
        filename: String,
        message: String,
    },

    #[error("IO error during file placement: {0}")]
    FilePlacement(String),
}

/// Callback-based download progress event.
#[derive(Debug, Clone)]
pub enum DownloadProgressEvent {
    FileStart {
        filename: String,
        file_index: usize,
        total_files: usize,
        size_bytes: u64,
    },
    FileProgress {
        filename: String,
        file_index: usize,
        bytes_downloaded: u64,
        bytes_total: u64,
    },
    FileDone {
        filename: String,
        file_index: usize,
        total_files: usize,
    },
}

pub type DownloadProgressCallback = Arc<dyn Fn(DownloadProgressEvent) + Send + Sync>;

// ── HF token resolution ─────────────────────────────────────────────────────

fn resolve_hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }
    Cache::new(hf_cache_dir())
        .token()
        .or_else(|| Cache::from_env().token())
}

// ── File placement ──────────────────────────────────────────────────────────

/// Hardlink `src` to `dst`, falling back to copy if hardlink fails (cross-filesystem).
/// Idempotent: skips if `dst` already exists with the same size as `src`.
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
            "failed to copy {} → {}: {e}",
            real_src.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

// ── SHA-256 verification ────────────────────────────────────────────────────

/// Verify the SHA-256 digest of a file against an expected hex string.
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let digest = format!("{:x}", hasher.finalize());
    Ok(digest == expected)
}

// ── Progress bar helpers ────────────────────────────────────────────────────

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

// ── Callback progress adapter ───────────────────────────────────────────────

#[derive(Clone)]
struct CallbackProgress {
    callback: DownloadProgressCallback,
    file_index: usize,
    total_files: usize,
    accumulated: u64,
    total: u64,
    filename: String,
    last_emit: Instant,
}

impl CallbackProgress {
    fn new(callback: DownloadProgressCallback, file_index: usize, total_files: usize) -> Self {
        Self {
            callback,
            file_index,
            total_files,
            accumulated: 0,
            total: 0,
            filename: String::new(),
            last_emit: Instant::now(),
        }
    }
}

impl Progress for CallbackProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.total = size as u64;
        self.accumulated = 0;
        self.filename = filename.to_string();
        (self.callback)(DownloadProgressEvent::FileStart {
            filename: self.filename.clone(),
            file_index: self.file_index,
            total_files: self.total_files,
            size_bytes: self.total,
        });
    }

    async fn update(&mut self, size: usize) {
        self.accumulated += size as u64;
        let now = Instant::now();
        if now.duration_since(self.last_emit).as_millis() >= 250 || self.accumulated >= self.total {
            self.last_emit = now;
            (self.callback)(DownloadProgressEvent::FileProgress {
                filename: self.filename.clone(),
                file_index: self.file_index,
                bytes_downloaded: self.accumulated,
                bytes_total: self.total,
            });
        }
    }

    async fn finish(&mut self) {
        (self.callback)(DownloadProgressEvent::FileDone {
            filename: self.filename.clone(),
            file_index: self.file_index,
            total_files: self.total_files,
        });
    }
}

// ── Core download function ──────────────────────────────────────────────────

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
            if status == Some(401) || err_str.contains("401") || err_str.contains("Unauthorized") {
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

// ── Public API ──────────────────────────────────────────────────────────────

/// Download all files for a model manifest with terminal progress bars.
///
/// Returns a list of `(component, clean_path)` pairs. The caller maps
/// these to their domain-specific model paths struct.
pub async fn pull_model<C: Component>(
    manifest: &ModelManifest<C>,
) -> Result<Vec<(C, PathBuf)>, DownloadError> {
    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
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

/// Download all files with callback-based progress (for non-terminal contexts).
pub async fn pull_model_with_callback<C: Component>(
    manifest: &ModelManifest<C>,
    callback: DownloadProgressCallback,
) -> Result<Vec<(C, PathBuf)>, DownloadError> {
    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let mdir = default_models_dir();
    let mut downloads: Vec<(C, PathBuf)> = Vec::new();
    let total_files = manifest.files.len();

    for (idx, file) in manifest.files.iter().enumerate() {
        let progress = CallbackProgress::new(callback.clone(), idx, total_files);

        let hf_path =
            download_file(&api, &file.hf_repo, &file.hf_filename, progress).await?;

        let clean_rel = storage_path(manifest, file);
        let clean_path = mdir.join(&clean_rel);
        hardlink_or_copy(&hf_path, &clean_path)?;

        if let Some(expected) = file.sha256 {
            match verify_sha256(&clean_path, expected) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!(
                        "warning: SHA-256 mismatch for {}",
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

/// Download a single file (sync). Safe to call from `spawn_blocking`.
/// If `target_subdir` is provided, hardlinks to `<models_dir>/<target_subdir>/<leaf>`.
pub fn download_single_file_sync(
    hf_repo: &str,
    hf_filename: &str,
    target_subdir: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| DownloadError::SyncApiSetup(e.to_string()))?;
    let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    let hf_path = repo.get(hf_filename).map_err(|e| {
        let err_str = e.to_string();
        if err_str.contains("401") || err_str.contains("Unauthorized") {
            DownloadError::Unauthorized {
                repo: hf_repo.to_string(),
            }
        } else if err_str.contains("403") || err_str.contains("Forbidden") || err_str.contains("gated") {
            DownloadError::GatedModel {
                repo: hf_repo.to_string(),
            }
        } else {
            DownloadError::SyncDownloadFailed {
                repo: hf_repo.to_string(),
                filename: hf_filename.to_string(),
                message: err_str,
            }
        }
    })?;

    if let Some(subdir) = target_subdir {
        let leaf = hf_filename.rsplit('/').next().unwrap_or(hf_filename);
        let clean_path = default_models_dir().join(subdir).join(leaf);
        hardlink_or_copy(&hf_path, &clean_path)?;
        Ok(clean_path)
    } else {
        Ok(hf_path)
    }
}

/// Check if a file is already cached locally (no download).
pub fn cached_file_path(hf_repo: &str, hf_filename: &str, target_subdir: Option<&str>) -> Option<PathBuf> {
    if let Some(subdir) = target_subdir {
        let leaf = hf_filename.rsplit('/').next().unwrap_or(hf_filename);
        let clean_path = default_models_dir().join(subdir).join(leaf);
        if clean_path.exists() {
            return Some(clean_path);
        }
    }

    let new_cache = Cache::new(hf_cache_dir());
    let new_repo = new_cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    if let Some(path) = new_repo.get(hf_filename) {
        return Some(path);
    }

    let default_cache = Cache::from_env();
    let default_repo = default_cache.repo(Repo::new(hf_repo.to_string(), RepoType::Model));
    default_repo.get(hf_filename)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_name_unchanged() {
        assert_eq!(truncate_filename("ae.safetensors", 45), "ae.safetensors");
    }

    #[test]
    fn truncate_long_name_keeps_suffix() {
        let result = truncate_filename("unet/diffusion_pytorch_model.fp16.safetensors", 30);
        assert_eq!(result.len(), 30);
        assert!(result.starts_with("..."));
        assert!(result.ends_with(".fp16.safetensors"));
    }

    #[test]
    fn truncate_very_small_max_returns_original() {
        let name = "something.safetensors";
        assert_eq!(truncate_filename(name, 5), name);
    }

    #[test]
    fn verify_sha256_matches() {
        let dir = std::env::temp_dir().join("lih_test_sha256_match");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_sha256(&path, expected).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_sha256_mismatch() {
        let dir = std::env::temp_dir().join("lih_test_sha256_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!verify_sha256(&path, wrong).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
