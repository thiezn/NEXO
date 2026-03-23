use local_inference_helpers::manifest::{Component, ManifestDefaults, ModelFile, ModelManifest};
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperComponent {
    Model,
    Tokenizer,
    Config,
    MelFilters,
}

impl Component for WhisperComponent {
    fn name(&self) -> &str {
        match self {
            Self::Model => "model",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::MelFilters => "mel_filters",
        }
    }

    fn is_model_specific(&self) -> bool {
        matches!(self, Self::Model)
    }
}

/// Unused ManifestDefaults (image-oriented fields) — zeroed for audio models.
const AUDIO_DEFAULTS: ManifestDefaults = ManifestDefaults {
    steps: 0,
    guidance: 0.0,
    width: 0,
    height: 0,
};

/// Build tokenizer + config files for a given HF repo.
fn metadata_files(hf_repo: &str, tokenizer_size: u64, config_size: u64) -> Vec<ModelFile<WhisperComponent>> {
    vec![
        ModelFile {
            component: WhisperComponent::Tokenizer,
            hf_repo: hf_repo.to_string(),
            hf_filename: "tokenizer.json".to_string(),
            size_bytes: tokenizer_size,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: WhisperComponent::Config,
            hf_repo: hf_repo.to_string(),
            hf_filename: "config.json".to_string(),
            size_bytes: config_size,
            gated: false,
            sha256: None,
        },
    ]
}

fn build_manifest(
    name: &str,
    description: &str,
    hf_repo: &str,
    model_size_bytes: u64,
    size_gb: f32,
    tokenizer_size: u64,
    config_size: u64,
) -> ModelManifest<WhisperComponent> {
    let mut files = metadata_files(hf_repo, tokenizer_size, config_size);
    files.push(ModelFile {
        component: WhisperComponent::Model,
        hf_repo: hf_repo.to_string(),
        hf_filename: "model.safetensors".to_string(),
        size_bytes: model_size_bytes,
        gated: false,
        sha256: None,
    });
    ModelManifest {
        name: name.to_string(),
        family: "whisper-v3".to_string(),
        description: description.to_string(),
        size_gb,
        files,
        defaults: AUDIO_DEFAULTS,
    }
}

// ── Model manifests ─────────────────────────────────────────────────────────

static WHISPER_LARGE_V3: LazyLock<ModelManifest<WhisperComponent>> = LazyLock::new(|| {
    build_manifest(
        "whisper-large-v3",
        "OpenAI Whisper Large V3 (32 encoder + 32 decoder layers, 1550M params)",
        "openai/whisper-large-v3",
        3_087_130_976,
        3.1,
        2_480_617,
        1_272,
    )
});

static WHISPER_LARGE_V3_TURBO: LazyLock<ModelManifest<WhisperComponent>> = LazyLock::new(|| {
    build_manifest(
        "whisper-large-v3-turbo",
        "OpenAI Whisper Large V3 Turbo (32 encoder + 4 decoder layers, 809M params)",
        "openai/whisper-large-v3-turbo",
        1_617_824_864,
        1.6,
        2_710_337,
        1_256,
    )
});

static DISTIL_LARGE_V3: LazyLock<ModelManifest<WhisperComponent>> = LazyLock::new(|| {
    build_manifest(
        "distil-large-v3",
        "Distil-Whisper Large V3 (distilled, 756M params, 6x faster)",
        "distil-whisper/distil-large-v3",
        1_512_874_472,
        1.5,
        2_480_617,
        1_372,
    )
});

pub fn known_manifests() -> Vec<&'static ModelManifest<WhisperComponent>> {
    vec![
        &WHISPER_LARGE_V3,
        &WHISPER_LARGE_V3_TURBO,
        &DISTIL_LARGE_V3,
    ]
}

pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<WhisperComponent>> {
    let canonical = resolve_model_name(name);
    known_manifests()
        .into_iter()
        .find(|m| m.name == canonical)
}

/// Resolve aliases and shorthand model names to canonical names.
pub fn resolve_model_name(name: &str) -> String {
    match name {
        "large-v3" | "v3" => "whisper-large-v3".to_string(),
        "turbo" | "large-v3-turbo" | "v3-turbo" => "whisper-large-v3-turbo".to_string(),
        "distil" | "distil-v3" | "distil-large-v3" => "distil-large-v3".to_string(),
        other => other.to_string(),
    }
}

pub fn total_download_size(manifest: &ModelManifest<WhisperComponent>) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}
