use super::manifest::{Component, ModelManifest, storage_path};
use super::paths::{default_models_dir, hf_cache_dir};

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use thiserror::Error;
use tracing::debug;

/// Errors that can occur during model download.
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error(
        "Model requires access approval on HuggingFace.\n\n  \
        1. Visit: https://hf-mirror/{repo}\n  \
        2. Accept the license agreement\n  \
        3. Create a token at: https://hf-mirror/settings/tokens\n  \
        4. Set: export HF_TOKEN=hf_...\n  \
        5. Retry the pull command"
    )]
    GatedModel { repo: String },

    #[error(
        "Authentication required for repository {repo}.\n\n  \
        1. Create a token at: https://hf-mirror/settings/tokens\n     \
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

/// Resolve a HuggingFace token from environment variable or cache file.
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
    Cache::new(hf_cache_dir())
        .token()
        .or_else(|| Cache::from_env().token())
}

/// Attempt to hardlink the source file to the destination. If that fails (e.g. across filesystems),
/// fall back to copying. If the destination already exists and has the same size as the source, do nothing.
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

/// Verify the SHA-256 hash of a file against an expected hex string.
pub fn verify_sha256(path: &std::path::Path, expected: &str) -> anyhow::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let digest = hasher.finalize();
    let digest = hex_encode(digest.as_ref());
    Ok(digest == expected)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Truncate long filenames for display in the progress bar, keeping the end of the name which often contains important info (e.g. version or hash).
fn truncate_filename(name: &str, max_len: usize) -> String {
    if name.len() <= max_len || max_len < 8 {
        return name.to_string();
    }
    let suffix_len = max_len - 3;
    let start = name.len() - suffix_len;
    format!("...{}", &name[start..])
}

/// Calculate the width for the filename column in the progress bar based on terminal size, leaving room for the bar and other info.
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

/// Extract HTTP status code from an ApiError if available, to help determine the cause of download failures.
fn extract_http_status(err: &ApiError) -> Option<u16> {
    if let ApiError::RequestError(reqwest_err) = err {
        reqwest_err.status().map(|s| s.as_u16())
    } else {
        None
    }
}

/// Download a file from HuggingFace with progress reporting, and handle common error cases to provide user-friendly messages.
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

/// Download all files for a model manifest with terminal progress bars.
///
/// Returns a list of `(component, clean_path)` pairs. The caller maps
/// these to their domain-specific model paths struct.
pub async fn pull_model<C: Component>(
    manifest: &ModelManifest<C>,
    force: bool,
) -> Result<Vec<(C, PathBuf)>, DownloadError> {
    let mdir = default_models_dir();
    let mut downloads: Vec<(C, PathBuf)> = Vec::new();
    let mut files_to_download = Vec::new();

    for file in &manifest.files {
        let clean_path = mdir.join(storage_path(manifest, file));
        if !force && std::fs::metadata(&clean_path).is_ok_and(|m| m.len() == file.size_bytes) {
            eprintln!(
                "  skipping {} (already downloaded, {:.1} MB)",
                file.hf_filename,
                file.size_bytes as f64 / 1_000_000.0
            );
            downloads.push((file.component.clone(), clean_path));
        } else {
            files_to_download.push(file);
        }
    }

    if files_to_download.is_empty() {
        return Ok(downloads);
    }

    eprintln!(
        "{} files to download for {}",
        files_to_download.len(),
        manifest.name
    );
    unsafe {
        std::env::set_var("HF_ENDPOINT", "https://hf-mirror.com");
    }
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

    for file in files_to_download {
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
