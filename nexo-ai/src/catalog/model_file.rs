/// A single file to download from Hugging Face.
#[derive(Debug, Clone)]
pub struct ModelFile {
    /// The kind of file
    pub kind: ModelFileKind,

    /// Hugging Face model repository.
    pub hf_repo: String,

    /// Path selector inside the Hugging Face repository.
    pub selector: ModelFileSelector,

    /// Optional clean path under this manifest's local storage directory.
    pub local_path: Option<String>,

    /// Expected file size in bytes, when known.
    ///
    /// It's a rough estimate, used for calculating the total required download for a model
    pub size_bytes: u64,

    /// Whether the source repository is gated.
    ///
    /// This happens when the model publisher restricts
    /// access to the repository, for example by requiring
    /// users to accept a license on Hugging Face before downloading.
    pub gated: bool,

    /// Expected SHA-256 hex digest. `None` means no digest is pinned yet.
    pub sha256: Option<&'static str>,
}

impl ModelFile {
    /// Returns the local path for this file, relative to the manifest's storage folder.
    pub fn local_path(&self) -> &str {
        self.local_path
            .as_deref()
            .unwrap_or_else(|| self.selector.label())
    }

    /// Checks if this model file has already been downloaded and verified locally.
    pub fn is_downloaded(&self) -> bool {
        todo!("Implement check if this ModelFile has been downloaded and verified locally")
    }

    /// Downloads this model file, returning an error if the download or verification fails.
    pub fn download(&self) -> Result {
        todo!("Implement download of this ModelFile, with verification and error handling")
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
    Exact(String),
    /// Every remote path ending with this suffix.
    Suffix(String),
    /// Every remote path starting with this prefix.
    Prefix(String),
}

impl ModelFileSelector {
    /// Returns whether a remote filename is matched by this selector.
    ///
    /// # Arguments
    ///
    /// * `filename` - The remote filename to check against this selector.
    pub fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(exact) => filename == exact,
            Self::Suffix(suffix) => filename.ends_with(suffix),
            Self::Prefix(prefix) => filename.starts_with(prefix),
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
