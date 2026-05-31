//! Hugging Face-backed model downloads into NEXO's local model store.

use std::io::Read;
use std::path::{Path, PathBuf};

use console::Term;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::task::JoinSet;
use tracing::debug;

use crate::manifest::{ModelComponent, ModelFile, ModelManifest, storage_path};
use crate::paths::{default_models_dir, hf_cache_dir};

/// Errors that can occur during model download.
#[derive(Debug, Error)]
pub enum DownloadError {
    /// The source repository is gated and requires approval.
    #[error("model requires access approval on Hugging Face: {repo}")]
    GatedModel {
        /// The gated Hugging Face repository.
        repo: String,
    },
    /// The source repository requires authentication.
    #[error("authentication required for Hugging Face repository {repo}")]
    Unauthorized {
        /// The Hugging Face repository that rejected unauthenticated access.
        repo: String,
    },
    /// A file failed to download.
    #[error("download failed for {filename} from {repo}: {source}")]
    DownloadFailed {
        /// The Hugging Face repository containing the file.
        repo: String,
        /// The repository-relative filename that failed to download.
        filename: String,
        /// The underlying Hugging Face API error.
        source: ApiError,
    },
    /// A pattern selector did not match any remote files.
    #[error("no files in {repo} matched selector {selector}")]
    NoFilesMatched {
        /// The Hugging Face repository being inspected.
        repo: String,
        /// The selector that matched no files.
        selector: String,
    },
    /// Hugging Face API client setup failed.
    #[error("failed to build Hugging Face API client: {0}")]
    ApiSetup(#[from] ApiError),
    /// Placing a downloaded file into the clean local model directory failed.
    #[error("failed to place downloaded file: {0}")]
    FilePlacement(String),
    /// SHA-256 verification could not read the file.
    #[error("failed to verify SHA-256 for {path}: {source}")]
    VerifyHash {
        /// The local file path being verified.
        path: PathBuf,
        /// The underlying file read error.
        source: std::io::Error,
    },
}

#[derive(Debug, Clone)]
struct ResolvedModelFile {
    component: ModelComponent,
    hf_repo: String,
    hf_filename: String,
    size_bytes: Option<u64>,
    sha256: Option<&'static str>,
}

#[derive(Debug, Clone)]
struct PendingDownload {
    component: ModelComponent,
    hf_repo: String,
    hf_filename: String,
    clean_path: PathBuf,
    sha256: Option<&'static str>,
    bar: ProgressBar,
}

const MAX_PARALLEL_DOWNLOADS: usize = 4;

/// Download all files for a model manifest with terminal progress bars.
pub async fn pull_model(
    manifest: &ModelManifest,
    force: bool,
) -> std::result::Result<Vec<(ModelComponent, PathBuf)>, DownloadError> {
    let models_dir = default_models_dir();
    let mut downloads = Vec::new();

    let mut builder = ApiBuilder::from_env().with_cache_dir(hf_cache_dir());
    // .with_endpoint(hf_endpoint()); // Skipping default mirror for now.
    if let Some(token) = resolve_hf_token() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;

    let mut resolved_files = Vec::new();
    for file in &manifest.files {
        resolved_files.extend(resolve_model_file(&api, file).await?);
    }

    let mut files_to_download = Vec::new();
    for file in resolved_files {
        let clean_path = models_dir.join(storage_path(manifest, &file.hf_filename));
        if !force && file_is_present(&clean_path, file.size_bytes) {
            eprintln!(
                "  skipping {} (already downloaded{})",
                file.hf_filename,
                size_suffix(file.size_bytes)
            );
            downloads.push((file.component, clean_path));
        } else {
            files_to_download.push((file, clean_path));
        }
    }

    if files_to_download.is_empty() {
        return Ok(downloads);
    }

    eprintln!(
        "{} files to download for {}",
        files_to_download.len(),
        manifest.id()
    );

    let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    let msg_width = filename_column_width();
    let bar_style = ProgressStyle::with_template(&format!(
        "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
    ))
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("#>-");

    let mut pending = files_to_download.into_iter().map(|(file, clean_path)| {
        let bar = multi.add(ProgressBar::new(file.size_bytes.unwrap_or_default()));
        bar.set_style(bar_style.clone());
        bar.set_message(truncate_filename(&file.hf_filename, msg_width));

        PendingDownload {
            component: file.component,
            hf_repo: file.hf_repo,
            hf_filename: file.hf_filename,
            clean_path,
            sha256: file.sha256,
            bar,
        }
    });

    let mut in_flight = JoinSet::new();
    for _ in 0..MAX_PARALLEL_DOWNLOADS {
        let Some(task) = pending.next() else {
            break;
        };
        spawn_download_task(&mut in_flight, api.clone(), task, msg_width);
    }

    while let Some(result) = in_flight.join_next().await {
        let (component, clean_path) = result
            .map_err(|error| DownloadError::FilePlacement(format!("download task failed: {error}")))??;
        downloads.push((component, clean_path));

        if let Some(task) = pending.next() {
            spawn_download_task(&mut in_flight, api.clone(), task, msg_width);
        }
    }

    Ok(downloads)
}

fn spawn_download_task(
    join_set: &mut JoinSet<std::result::Result<(ModelComponent, PathBuf), DownloadError>>,
    api: Api,
    task: PendingDownload,
    msg_width: usize,
) {
    join_set.spawn(async move {
        let hf_path = download_file(
            &api,
            &task.hf_repo,
            &task.hf_filename,
            DownloadProgress::new(task.bar, msg_width),
        )
        .await?;

        hardlink_or_copy(&hf_path, &task.clean_path)?;

        if let Some(expected) = task.sha256 {
            match verify_sha256(&task.clean_path, expected) {
                Ok(true) => {}
                Ok(false) => eprintln!(
                    "warning: SHA-256 mismatch for {} (file may have changed upstream)",
                    task.hf_filename
                ),
                Err(error) => eprintln!("warning: {error}"),
            }
        }

        Ok((task.component, task.clean_path))
    });
}

async fn resolve_model_file(
    api: &Api,
    file: &ModelFile,
) -> std::result::Result<Vec<ResolvedModelFile>, DownloadError> {
    if let Some(filename) = file.selector.exact_path() {
        return Ok(vec![ResolvedModelFile {
            component: file.component,
            hf_repo: file.hf_repo.clone(),
            hf_filename: filename.to_string(),
            size_bytes: file.size_bytes,
            sha256: file.sha256,
        }]);
    }

    let repo = api.repo(Repo::new(file.hf_repo.clone(), RepoType::Model));
    let info = repo
        .info()
        .await
        .map_err(|source| DownloadError::DownloadFailed {
            repo: file.hf_repo.clone(),
            filename: file.selector.label().to_string(),
            source,
        })?;

    let mut filenames = info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|filename| file.selector.matches(filename))
        .collect::<Vec<_>>();
    filenames.sort();

    if filenames.is_empty() {
        return Err(DownloadError::NoFilesMatched {
            repo: file.hf_repo.clone(),
            selector: file.selector.label().to_string(),
        });
    }

    Ok(filenames
        .into_iter()
        .map(|hf_filename| ResolvedModelFile {
            component: file.component,
            hf_repo: file.hf_repo.clone(),
            hf_filename,
            size_bytes: file.size_bytes,
            sha256: file.sha256,
        })
        .collect())
}

fn file_is_present(path: &Path, expected_size: Option<u64>) -> bool {
    std::fs::metadata(path)
        .map(|metadata| expected_size.is_none_or(|size| metadata.len() == size))
        .unwrap_or(false)
}

fn size_suffix(size_bytes: Option<u64>) -> String {
    size_bytes.map_or_else(String::new, |bytes| {
        format!(", {:.1} MB", bytes as f64 / 1_000_000.0)
    })
}

/// Verify the SHA-256 hash of a file against an expected hex string.
pub fn verify_sha256(path: &Path, expected: &str) -> std::result::Result<bool, DownloadError> {
    let mut file = std::fs::File::open(path).map_err(|source| DownloadError::VerifyHash {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|source| DownloadError::VerifyHash {
                path: path.to_path_buf(),
                source,
            })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex_encode(hasher.finalize().as_ref()) == expected)
}

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

#[allow(dead_code)]
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://hf-mirror.com".to_string())
}

fn hardlink_or_copy(src: &Path, dst: &Path) -> std::result::Result<(), DownloadError> {
    let real_src = src.canonicalize().map_err(|error| {
        DownloadError::FilePlacement(format!(
            "source file not found after download: {} ({error})",
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
        std::fs::create_dir_all(parent).map_err(|error| {
            DownloadError::FilePlacement(format!(
                "failed to create directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    if std::fs::hard_link(&real_src, dst).is_ok() {
        return Ok(());
    }

    std::fs::copy(&real_src, dst).map_err(|error| {
        DownloadError::FilePlacement(format!(
            "failed to copy {} -> {}: {error}",
            real_src.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

async fn download_file<P: Progress + Clone + Send + Sync + 'static>(
    api: &Api,
    hf_repo: &str,
    hf_filename: &str,
    progress: P,
) -> std::result::Result<PathBuf, DownloadError> {
    let repo = api.repo(Repo::new(hf_repo.to_string(), RepoType::Model));

    match repo.download_with_progress(hf_filename, progress).await {
        Ok(path) => Ok(path),
        Err(error) => match extract_http_status(&error) {
            Some(401) => Err(DownloadError::Unauthorized {
                repo: hf_repo.to_string(),
            }),
            Some(403) => Err(DownloadError::GatedModel {
                repo: hf_repo.to_string(),
            }),
            _ => Err(DownloadError::DownloadFailed {
                repo: hf_repo.to_string(),
                filename: hf_filename.to_string(),
                source: error,
            }),
        },
    }
}

fn extract_http_status(error: &ApiError) -> Option<u16> {
    if let ApiError::RequestError(request_error) = error {
        request_error.status().map(|status| status.as_u16())
    } else {
        None
    }
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

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
