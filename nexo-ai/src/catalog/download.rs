use crate::{Error, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

use super::paths::hf_cache_dir;

/// Default maximum number of file downloads running concurrently.
pub(crate) const DEFAULT_MAX_CONCURRENT_FILES: usize = 4;

/// Options controlling catalog download behavior.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Force a re-download even if the final file already validates locally.
    pub force: bool,

    /// Maximum number of files to download concurrently across one active download request.
    pub max_concurrent_files: usize,

    /// Remove the staged HF cache file after it has been copied into place.
    pub cleanup_cache_on_success: bool,
}

impl Default for DownloadOptions {
    /// Returns the default download behavior.
    fn default() -> Self {
        Self {
            force: false,
            max_concurrent_files: DEFAULT_MAX_CONCURRENT_FILES,
            cleanup_cache_on_success: true,
        }
    }
}

/// Shared downloader context for catalog-managed model files.
#[derive(Debug, Clone)]
pub(crate) struct CatalogDownloader {
    api: Api,
    options: DownloadOptions,
}

impl CatalogDownloader {
    /// Builds a downloader using the configured environment and cache path.
    ///
    /// # Arguments
    ///
    /// * `options` - Download settings controlling concurrency, force mode, and cache cleanup.
    pub(crate) fn new(options: DownloadOptions) -> Result<Self> {
        let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
        if let Some(token) = resolve_hf_token() {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build()?;
        Ok(Self { api, options })
    }

    /// Returns the configured download options.
    #[must_use]
    pub(crate) fn options(&self) -> &DownloadOptions {
        &self.options
    }

    /// Downloads a remote Hugging Face file into the local HF cache.
    ///
    /// # Arguments
    ///
    /// * `hf_repo` - The Hugging Face repository containing the file.
    /// * `remote_path` - The repository-relative path of the file to download.
    /// * `progress` - The per-file progress adapter used during the transfer.
    pub(crate) async fn download_to_cache<P>(
        &self,
        hf_repo: &str,
        remote_path: &str,
        progress: P,
    ) -> Result<PathBuf>
    where
        P: Progress + Clone + Send + Sync + 'static,
    {
        let repo = self
            .api
            .repo(Repo::new(hf_repo.to_string(), RepoType::Model));
        match repo.download_with_progress(remote_path, progress).await {
            Ok(path) => Ok(path),
            Err(error) => Err(map_download_error(hf_repo, remote_path, error)),
        }
    }

    /// Copies a staged cache file into its final destination.
    ///
    /// # Arguments
    ///
    /// * `cache_path` - The staged file path inside the Hugging Face cache.
    /// * `destination` - The final runtime-facing file path.
    pub(crate) fn copy_cache_file(&self, cache_path: &Path, destination: &Path) -> Result {
        copy_file_into_place(cache_path, destination)
    }

    /// Deletes a staged cache file after a successful final copy.
    ///
    /// # Arguments
    ///
    /// * `cache_path` - The staged file path that can be removed after success.
    pub(crate) fn cleanup_cache_file(&self, cache_path: &Path) -> Result {
        if cache_path.exists() {
            std::fs::remove_file(cache_path)?;
        }
        Ok(())
    }
}

/// Resolves the configured Hugging Face authentication token.
fn resolve_hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("Using Hugging Face token from HF_TOKEN");
            return Some(token);
        }
    }

    dirs::home_dir()
        .map(|home| home.join(".nexo").join("hf_token.txt"))
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .or_else(|| Cache::new(hf_cache_dir()).token())
        .or_else(|| Cache::from_env().token())
}

/// Maps an `hf_hub` API error into a contextual catalog download error.
///
/// # Arguments
///
/// * `hf_repo` - The Hugging Face repository that was being accessed.
/// * `remote_path` - The repository-relative file path involved in the failed request.
/// * `error` - The original `hf_hub` API error.
fn map_download_error(hf_repo: &str, remote_path: &str, error: ApiError) -> Error {
    match extract_http_status(&error) {
        Some(401) => Error::Unauthorized {
            repo: hf_repo.to_string(),
        },
        Some(403) => Error::GatedModel {
            repo: hf_repo.to_string(),
        },
        _ => Error::DownloadFailed {
            repo: hf_repo.to_string(),
            filename: remote_path.to_string(),
            source: error,
        },
    }
}

/// Extracts an HTTP status code from an `hf_hub` API error when present.
///
/// # Arguments
///
/// * `error` - The API error to inspect.
fn extract_http_status(error: &ApiError) -> Option<u16> {
    if let ApiError::RequestError(request_error) = error {
        request_error.status().map(|status| status.as_u16())
    } else {
        None
    }
}

/// Copies a file into its final destination using a temporary rename step.
///
/// # Arguments
///
/// * `src` - The staged source file to copy from.
/// * `dst` - The final destination path to replace atomically.
fn copy_file_into_place(src: &Path, dst: &Path) -> Result {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let temp_path = temp_destination_path(dst);
    if temp_path.exists() {
        let _ = std::fs::remove_file(&temp_path);
    }

    if let Err(error) = std::fs::copy(src, &temp_path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error.into());
    }

    if dst.exists() {
        std::fs::remove_file(dst)?;
    }

    if let Err(error) = std::fs::rename(&temp_path, dst) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error.into());
    }

    Ok(())
}

/// Creates a temporary sibling path for an in-progress destination file.
///
/// # Arguments
///
/// * `dst` - The final destination path.
fn temp_destination_path(dst: &Path) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = dst
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.download-{unique}"))
        .unwrap_or_else(|| format!("download-{unique}"));
    dst.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::{CatalogDownloader, DownloadOptions, copy_file_into_place};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nexo-ai-download-{name}-{unique}"))
    }

    #[test]
    fn default_options_cleanup_cache_after_success() {
        let options = DownloadOptions::default();
        assert!(options.cleanup_cache_on_success);
        assert_eq!(
            options.max_concurrent_files,
            super::DEFAULT_MAX_CONCURRENT_FILES
        );
    }

    #[test]
    fn copy_places_file_at_destination() {
        let dir = temp_dir("copy");
        fs::create_dir_all(&dir).expect("create temp dir");
        let src = dir.join("source.bin");
        let dst = dir.join("nested").join("dest.bin");
        fs::write(&src, b"hello world").expect("write source");

        copy_file_into_place(&src, &dst).expect("copy into place");

        assert_eq!(fs::read(&dst).expect("read destination"), b"hello world");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_removes_staged_cache_file() {
        let dir = temp_dir("cleanup");
        fs::create_dir_all(&dir).expect("create temp dir");
        let staged = dir.join("cached.bin");
        fs::write(&staged, b"cache").expect("write staged file");

        let downloader = CatalogDownloader::new(DownloadOptions::default())
            .expect("create downloader from environment");
        downloader
            .cleanup_cache_file(&staged)
            .expect("cleanup staged file");

        assert!(!staged.exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
