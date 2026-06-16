use crate::Result;

/// A single file to download from Hugging Face.
#[derive(Debug, Clone)]
pub struct ModelFile {
    /// The kind of file
    kind: ModelFileKind,

    /// Hugging Face model repository.
    hf_repo: &'static str,

    /// Path selector inside the Hugging Face repository.
    selector: ModelFileSelector,

    /// File size in bytes.
    ///
    /// Used for calculating the total required download for a model. If the ModelFileSelector
    /// is a suffix or prefix, this is the total size of all matching files.
    size_bytes: u64,

    /// Expected SHA-256 hex digest. `None` means no digest is pinned yet.
    sha256: &'static str,
}

impl ModelFile {
    /// Initialize new ModelFile with the given parameters.
    pub const fn new(
        kind: ModelFileKind,
        hf_repo: &'static str,
        filename: &'static str,
        size_bytes: u64,
        sha256: &'static str,
    ) -> Self {
        Self {
            kind,
            hf_repo,
            selector: ModelFileSelector::Exact(filename),
            size_bytes,
            sha256,
        }
    }

    /// Initialize new ModelFile with a filename suffix selector.
    ///
    /// Useful to match on any file with a certain suffix, that have
    /// the same ModelFileKind and are interchangeable in the manifest.
    pub fn new_with_suffix(
        kind: ModelFileKind,
        hf_repo: &'static str,
        suffix: &'static str,
        size_bytes: u64,
        sha256: &'static str,
    ) -> Self {
        Self {
            kind,
            hf_repo,
            selector: ModelFileSelector::Suffix(suffix),
            size_bytes,
            sha256,
        }
    }

    /// Initialize new ModelFile with a filename prefix selector.
    pub fn new_with_prefix(
        kind: ModelFileKind,
        hf_repo: &'static str,
        prefix: &'static str,
        size_bytes: u64,
        sha256: &'static str,
    ) -> Self {
        Self {
            kind,
            hf_repo,
            selector: ModelFileSelector::Prefix(prefix),
            size_bytes,
            sha256,
        }
    }

    /// Checks if this model file has already been downloaded and verified locally.
    pub fn is_downloaded(&self) -> bool {
        todo!("Implement check if this ModelFile has been downloaded and verified locally")
    }

    /// Downloads this model file, returning an error if the download or verification fails.
    pub fn download(&self) -> Result {
        todo!("Implement download of this ModelFile, with verification and error handling")
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
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
}

/// How a manifest selects files from a remote model repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFileSelector {
    /// A single known path in the remote repository.
    Exact(&'static str),
    /// Every remote path ending with this suffix.
    Suffix(&'static str),
    /// Every remote path starting with this prefix.
    Prefix(&'static str),
}

impl ModelFileSelector {
    /// Returns whether a remote filename is matched by this selector.
    ///
    /// # Arguments
    ///
    /// * `filename` - The remote filename to check against this selector.
    pub fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(exact) => filename == *exact,
            Self::Suffix(suffix) => filename.ends_with(*suffix),
            Self::Prefix(prefix) => filename.starts_with(*prefix),
        }
    }

    /// Returns the exact remote path when this selector is exact.
    pub fn exact_path(&self) -> Option<&str> {
        match self {
            Self::Exact(path) => Some(path),
            Self::Suffix(_) | Self::Prefix(_) => None,
        }
    }

    /// Human-readable selector label used in diagnostics.
    pub fn label(&self) -> &str {
        match self {
            Self::Exact(path) | Self::Suffix(path) | Self::Prefix(path) => path,
        }
    }
}
