use super::download::CatalogDownloader;
use super::paths::is_relative_storage_path;
use super::progress::HfHubProgressAdapter;
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A single file to download from Hugging Face.
#[derive(Debug, Clone)]
pub struct ModelFile {
    /// The kind of file
    kind: ModelFileKind,

    /// Hugging Face model repository.
    hf_repo: &'static str,

    /// Exact path of the file inside the Hugging Face repository.
    remote_path: &'static str,

    /// Optional local subfolder relative to the manifest storage folder.
    sub_folder: Option<&'static str>,

    /// File size in bytes.
    size_bytes: u64,

    /// Expected SHA-256 hex digest.
    sha256: &'static str,
}

impl ModelFile {
    /// Initialize new ModelFile with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `kind` - The semantic role of the file within the model artifact.
    /// * `hf_repo` - The Hugging Face repository containing the file.
    /// * `remote_path` - The exact repository-relative path of the file.
    /// * `size_bytes` - The expected size of the file in bytes.
    /// * `sha256` - The expected SHA-256 digest of the file.
    pub const fn new(
        kind: ModelFileKind,
        hf_repo: &'static str,
        remote_path: &'static str,
        size_bytes: u64,
        sha256: &'static str,
    ) -> Self {
        Self {
            kind,
            hf_repo,
            remote_path,
            sub_folder: None,
            size_bytes,
            sha256,
        }
    }

    /// Initialize a new ModelFile that is stored in a local subfolder.
    ///
    /// # Arguments
    ///
    /// * `kind` - The semantic role of the file within the model artifact.
    /// * `hf_repo` - The Hugging Face repository containing the file.
    /// * `remote_path` - The exact repository-relative path of the file.
    /// * `sub_folder` - The local subfolder under the manifest storage folder where the file should be placed.
    /// * `size_bytes` - The expected size of the file in bytes.
    /// * `sha256` - The expected SHA-256 digest of the file.
    pub const fn new_in_subfolder(
        kind: ModelFileKind,
        hf_repo: &'static str,
        remote_path: &'static str,
        sub_folder: &'static str,
        size_bytes: u64,
        sha256: &'static str,
    ) -> Self {
        Self {
            kind,
            hf_repo,
            remote_path,
            sub_folder: Some(sub_folder),
            size_bytes,
            sha256,
        }
    }

    /// Checks whether this model file has already been downloaded and verified locally.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The local model directory that should contain the final file.
    pub(crate) fn is_downloaded(&self, model_dir: &Path) -> bool {
        let Ok(local_path) = self.local_path(model_dir) else {
            return false;
        };

        if !local_path.is_file() || !has_valid_sha256(self.sha256) {
            return false;
        }

        self.verify_local_file(&local_path).is_ok()
    }

    /// Checks whether this model file is present locally using a fast metadata check.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The local model directory that should contain the final file.
    pub(crate) fn is_present(&self, model_dir: &Path) -> bool {
        let Ok(local_path) = self.local_path(model_dir) else {
            return false;
        };

        match std::fs::metadata(local_path) {
            Ok(metadata) if metadata.is_file() => metadata.len() == self.size_bytes,
            _ => false,
        }
    }

    /// Downloads this model file into the given model directory.
    ///
    /// # Arguments
    ///
    /// * `downloader` - The shared downloader context that owns the HF client and download options.
    /// * `model_dir` - The local model directory that should contain the final file.
    /// * `progress` - The per-file progress sink that receives byte updates during the transfer.
    pub(crate) async fn download(
        &self,
        downloader: &CatalogDownloader,
        model_dir: &Path,
        progress: Arc<dyn crate::FileDownloadProgress>,
    ) -> Result {
        let local_path = self.local_path(model_dir)?;

        if !downloader.options().force
            && local_path.is_file()
            && self.verify_local_file(&local_path).is_ok()
        {
            return Ok(());
        }

        let cache_path = downloader
            .download_to_cache(
                self.hf_repo(),
                self.remote_path(),
                HfHubProgressAdapter::new(progress, self.remote_path().to_string()),
            )
            .await?;

        self.verify_local_file(&cache_path)?;
        downloader.copy_cache_file(&cache_path, &local_path)?;
        self.verify_local_file(&local_path)?;

        if downloader.options().cleanup_cache_on_success {
            downloader.cleanup_cache_file(&cache_path)?;
        }

        Ok(())
    }

    /// Returns the size of this model file in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Returns the semantic kind of this model file.
    #[must_use]
    pub(crate) fn kind(&self) -> ModelFileKind {
        self.kind
    }

    /// Returns the Hugging Face repository that contains this file.
    #[must_use]
    pub(crate) fn hf_repo(&self) -> &'static str {
        self.hf_repo
    }

    /// Returns the exact repository-relative remote path for this file.
    #[must_use]
    pub(crate) fn remote_path(&self) -> &'static str {
        self.remote_path
    }

    /// Resolves the final local file path for this model file.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The base model directory for the owning manifest.
    pub(crate) fn local_path(&self, model_dir: &Path) -> Result<PathBuf> {
        if !is_relative_storage_path(Path::new(self.remote_path)) {
            return Err(Error::InvalidModelFilePath {
                path: self.remote_path.to_string(),
            });
        }

        let relative_path = match self.sub_folder {
            Some(sub_folder) => {
                let sub_folder = Path::new(sub_folder);
                if !is_relative_storage_path(sub_folder) {
                    return Err(Error::InvalidModelFilePath {
                        path: sub_folder.display().to_string(),
                    });
                }

                let Some(file_name) = Path::new(self.remote_path).file_name() else {
                    return Err(Error::InvalidModelFilePath {
                        path: self.remote_path.to_string(),
                    });
                };
                sub_folder.join(file_name)
            }
            None => PathBuf::from(self.remote_path),
        };

        Ok(model_dir.join(relative_path))
    }

    /// Verifies the SHA-256 digest of a local file against the configured manifest digest.
    ///
    /// # Arguments
    ///
    /// * `path` - The local file path to verify.
    pub(crate) fn verify_local_file(&self, path: &Path) -> Result {
        if !has_valid_sha256(self.sha256) {
            return Err(Error::InvalidConfiguredSha256 {
                repo: self.hf_repo.to_string(),
                remote_path: self.remote_path.to_string(),
                sha256: self.sha256.to_string(),
            });
        }

        let actual = compute_sha256(path)?;
        if actual != self.sha256 {
            return Err(Error::Sha256Mismatch {
                expected: self.sha256.to_string(),
                actual,
                repo: self.hf_repo.to_string(),
                remote_path: self.remote_path.to_string(),
                local_path: path.to_path_buf(),
            });
        }

        Ok(())
    }
}

/// Component types for files that make up a model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFileKind {
    /// Primary model weight file.
    Weights,
    /// Sharded model weight file.
    WeightShard,
    /// Mistral UQFF quantized artifact.
    UqffShard,
    /// Non-quantized residual tensors needed next to UQFF artifacts.
    UqffResidual,
    /// Tokenizer file.
    Tokenizer,
    /// Tokenizer sidecar configuration.
    TokenizerConfig,
    /// Model configuration or auxiliary metadata file.
    Config,
    /// Generation defaults.
    GenerationConfig,
    /// Chat template file.
    ChatTemplate,
    /// Multimodal processor configuration.
    ProcessorConfig,
    /// Multimodal preprocessor configuration.
    PreprocessorConfig,
    /// Embedding module manifest.
    Modules,
    /// Multimodal projector weights.
    VisionProjector,
    /// Audio Voice files used in some TTS models.
    Voice,
}

/// Returns whether a configured SHA-256 string is syntactically valid.
///
/// # Arguments
///
/// * `sha256` - The configured digest string to validate.
fn has_valid_sha256(sha256: &str) -> bool {
    sha256.len() == 64 && sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Computes the SHA-256 digest of a file.
///
/// # Arguments
///
/// * `path` - The file path to hash.
fn compute_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex_encode(hasher.finalize().as_ref()))
}

/// Encodes raw bytes as lowercase hexadecimal.
///
/// # Arguments
///
/// * `bytes` - The bytes to encode as lowercase hexadecimal.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{ModelFile, ModelFileKind};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nexo-ai-{name}-{unique}"))
    }

    #[test]
    fn exact_file_is_downloaded_when_hash_matches() {
        let dir = temp_dir("exact-match");
        fs::create_dir_all(&dir).expect("create temp dir");
        let file_path = dir.join("weights.bin");
        fs::write(&file_path, b"hello world").expect("write test file");

        let file = ModelFile::new(
            ModelFileKind::Weights,
            "repo",
            "weights.bin",
            11,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        );

        assert!(file.is_downloaded(&dir));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exact_file_is_not_downloaded_when_hash_is_placeholder() {
        let dir = temp_dir("placeholder-hash");
        fs::create_dir_all(&dir).expect("create temp dir");
        let file_path = dir.join("weights.bin");
        fs::write(&file_path, b"hello world").expect("write test file");

        let file = ModelFile::new(
            ModelFileKind::Weights,
            "repo",
            "weights.bin",
            11,
            "placeholder_sha256",
        );

        assert!(!file.is_downloaded(&dir));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn subfolder_files_use_remote_basename_locally() {
        let dir = temp_dir("sub-folder");
        let voices_dir = dir.join("voices");
        fs::create_dir_all(&voices_dir).expect("create temp dir");
        let file_path = voices_dir.join("voice.pt");
        fs::write(&file_path, b"subfolder file").expect("write test file");

        let file = ModelFile::new_in_subfolder(
            ModelFileKind::Voice,
            "repo",
            "remote/voice.pt",
            "voices",
            14,
            "f789d539e8fa8c962e25e3a7f2ce5c5890ffa53d781a286adecf073de71b2597",
        );

        assert!(file.is_downloaded(&dir));

        let _ = fs::remove_dir_all(&dir);
    }
}
