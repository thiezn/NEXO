use super::manifest::{Component, ModelManifest, storage_path};
use super::paths::{default_models_dir, nexo_home_dir};

use console::Term;
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use thiserror::Error;
use tracing::debug;

/// Errors that can occur during model download.
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error(
        "Model requires access approval on HuggingFace.\n\n  \
        1. Visit the model page on HuggingFace and accept the license agreement\n  \
        2. Create a token at: https://huggingface.co/settings/tokens\n  \
        3. Set: export HF_TOKEN=hf_...\n  \
        4. Retry the pull command"
    )]
    GatedModel { repo: String },

    #[error(
        "Authentication required for repository {repo}.\n\n  \
        1. Create a token at: https://huggingface.co/settings/tokens\n  \
        2. Set: export HF_TOKEN=hf_...\n  \
        3. Retry the pull command"
    )]
    Unauthorized { repo: String },

    #[error("Download failed for {filename} from {repo}: {message}")]
    DownloadFailed {
        repo: String,
        filename: String,
        message: String,
    },

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
    let token_path = nexo_home_dir().join("hf_token.txt");
    if let Ok(token) = std::fs::read_to_string(&token_path) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("Using HuggingFace token from {}", token_path.display());
            return Some(token);
        }
    }
    None
}

fn hf_base_url() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://hf-mirror.com".to_string())
}

fn build_download_url(hf_repo: &str, hf_filename: &str) -> String {
    let base = hf_base_url();
    format!("{base}/{hf_repo}/resolve/main/{hf_filename}")
}

// ── File placement ──────────────────────────────────────────────────

fn move_or_copy(src: &std::path::Path, dst: &std::path::Path) -> Result<(), DownloadError> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            DownloadError::FilePlacement(format!(
                "failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }

    // Try atomic rename first; falls back to copy if on different filesystems.
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }

    std::fs::copy(src, dst).map_err(|e| {
        DownloadError::FilePlacement(format!(
            "failed to copy {} -> {}: {e}",
            src.display(),
            dst.display()
        ))
    })?;
    let _ = std::fs::remove_file(src);
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

// ── Core download function ──────────────────────────────────────────

async fn download_file(
    client: &reqwest::Client,
    hf_repo: &str,
    hf_filename: &str,
    dest: &std::path::Path,
    bar: ProgressBar,
) -> Result<(), DownloadError> {
    let url = build_download_url(hf_repo, hf_filename);
    debug!("Downloading {url}");

    // Write to a temp file first so we never leave a partial file at dest.
    let tmp_path = dest.with_extension("tmp");
    if let Some(parent) = tmp_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            DownloadError::FilePlacement(format!(
                "failed to create directory: {e}"
            ))
        })?;
    }

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| DownloadError::DownloadFailed {
            repo: hf_repo.to_string(),
            filename: hf_filename.to_string(),
            message: e.to_string(),
        })?;

    let status = response.status();
    if status == 401 {
        return Err(DownloadError::Unauthorized {
            repo: hf_repo.to_string(),
        });
    }
    if status == 403 {
        return Err(DownloadError::GatedModel {
            repo: hf_repo.to_string(),
        });
    }
    if !status.is_success() {
        return Err(DownloadError::DownloadFailed {
            repo: hf_repo.to_string(),
            filename: hf_filename.to_string(),
            message: format!("HTTP {status}"),
        });
    }

    if let Some(content_length) = response.content_length() {
        bar.set_length(content_length);
    }

    let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
        DownloadError::FilePlacement(format!("failed to create temp file: {e}"))
    })?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e: reqwest::Error| DownloadError::DownloadFailed {
            repo: hf_repo.to_string(),
            filename: hf_filename.to_string(),
            message: e.to_string(),
        })?;
        std::io::Write::write_all(&mut file, &chunk).map_err(|e| {
            DownloadError::FilePlacement(format!("failed to write chunk: {e}"))
        })?;
        bar.inc(chunk.len() as u64);
    }

    bar.finish_with_message("done");
    drop(file);

    move_or_copy(&tmp_path, dest)
}

// ── Public API ──────────────────────────────────────────────────────

/// Download all files for a model manifest with terminal progress bars.
///
/// For existing files:
/// - If `sha256` is set in the manifest, verifies the hash. Re-downloads on mismatch.
/// - Otherwise falls back to size comparison.
///
/// Returns a list of `(component, clean_path)` pairs.
pub async fn pull_model<C: Component>(
    manifest: &ModelManifest<C>,
    force: bool,
) -> Result<Vec<(C, PathBuf)>, DownloadError> {
    let mdir = default_models_dir();
    let mut downloads: Vec<(C, PathBuf)> = Vec::new();
    let mut files_to_download = Vec::new();

    for file in &manifest.files {
        let clean_path = mdir.join(storage_path(manifest, file));

        let (already_valid, sha_mismatch) = match file.sha256 {
            Some(expected) => match verify_sha256(&clean_path, expected) {
                Ok(true) => (true, false),
                Ok(false) => (false, true),
                Err(_) => (false, false), // file absent or unreadable
            },
            None => {
                // No SHA to verify — fall back to size check, or just existence
                // if the manifest has no known size (size_bytes == 0).
                let valid = std::fs::metadata(&clean_path).is_ok_and(|m| {
                    if file.size_bytes > 0 {
                        m.len() == file.size_bytes
                    } else {
                        m.len() > 0
                    }
                });
                (valid, false)
            }
        };

        if !force && already_valid {
            eprintln!(
                "  skipping {} (already downloaded and verified)",
                file.hf_filename
            );
            downloads.push((file.component.clone(), clean_path));
        } else {
            if sha_mismatch {
                eprintln!("  re-downloading {} (SHA-256 mismatch)", file.hf_filename);
            }
            files_to_download.push(file);
        }
    }

    if files_to_download.is_empty() {
        return Ok(downloads);
    }

    eprintln!(
        "{} file(s) to download for {} (via {})",
        files_to_download.len(),
        manifest.name,
        hf_base_url(),
    );

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(token) = resolve_hf_token() {
        let val = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .expect("token should be ASCII");
        headers.insert(reqwest::header::AUTHORIZATION, val);
    }

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| DownloadError::DownloadFailed {
            repo: manifest.name.clone(),
            filename: String::new(),
            message: format!("failed to build HTTP client: {e}"),
        })?;

    let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    let msg_width = filename_column_width();
    let bar_style = ProgressStyle::with_template(&format!(
        "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
    ))
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("━╸─");

    for file in files_to_download {
        let clean_path = mdir.join(storage_path(manifest, file));
        let bar = multi.add(ProgressBar::new(file.size_bytes));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        download_file(&client, &file.hf_repo, &file.hf_filename, &clean_path, bar).await?;

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
