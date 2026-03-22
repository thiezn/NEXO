use local_inference_helpers::manifest::{Component, ManifestDefaults, ModelFile, ModelManifest};
use std::collections::HashMap;
use std::sync::LazyLock;

/// TTS-specific model component types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TTSComponent {
    Decoder,
    DecoderShard,
    Tokenizer,
    Config,
}

impl Component for TTSComponent {
    fn name(&self) -> &str {
        match self {
            Self::Decoder => "decoder",
            Self::DecoderShard => "decoder_shard",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
        }
    }

    fn is_model_specific(&self) -> bool {
        matches!(self, Self::Decoder | Self::DecoderShard)
    }
}

// ── TTS defaults (not applicable fields zeroed) ─────────────────────────────

const TTS_DEFAULTS: ManifestDefaults = ManifestDefaults {
    steps: 0,
    guidance: 0.0,
    width: 0,
    height: 0,
};

// ── Parler-TTS manifests ────────────────────────────────────────────────────

fn parler_mini_manifest() -> ModelManifest<TTSComponent> {
    ModelManifest {
        name: "parler-mini".to_string(),
        family: "parler".to_string(),
        description: "Parler-TTS Mini v1.1 — fast, lightweight TTS (~2.2 GB)".to_string(),
        size_gb: 2.2,
        files: vec![
            ModelFile {
                component: TTSComponent::Decoder,
                hf_repo: "parler-tts/parler-tts-mini-v1.1".to_string(),
                hf_filename: "model.safetensors".to_string(),
                size_bytes: 2_200_000_000,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: TTSComponent::Config,
                hf_repo: "parler-tts/parler-tts-mini-v1.1".to_string(),
                hf_filename: "config.json".to_string(),
                size_bytes: 3_000,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: TTSComponent::Tokenizer,
                hf_repo: "parler-tts/parler-tts-mini-v1.1".to_string(),
                hf_filename: "tokenizer.json".to_string(),
                size_bytes: 2_500_000,
                gated: false,
                sha256: None,
            },
        ],
        defaults: TTS_DEFAULTS,
    }
}

fn parler_large_manifest() -> ModelManifest<TTSComponent> {
    ModelManifest {
        name: "parler-large".to_string(),
        family: "parler".to_string(),
        description: "Parler-TTS Large v1 — higher quality TTS (~9.4 GB)".to_string(),
        size_gb: 9.4,
        files: vec![
            ModelFile {
                component: TTSComponent::DecoderShard,
                hf_repo: "parler-tts/parler-tts-large-v1".to_string(),
                hf_filename: "model-00001-of-00002.safetensors".to_string(),
                size_bytes: 4_900_000_000,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: TTSComponent::DecoderShard,
                hf_repo: "parler-tts/parler-tts-large-v1".to_string(),
                hf_filename: "model-00002-of-00002.safetensors".to_string(),
                size_bytes: 4_500_000_000,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: TTSComponent::Config,
                hf_repo: "parler-tts/parler-tts-large-v1".to_string(),
                hf_filename: "config.json".to_string(),
                size_bytes: 3_000,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: TTSComponent::Tokenizer,
                hf_repo: "parler-tts/parler-tts-large-v1".to_string(),
                hf_filename: "tokenizer.json".to_string(),
                size_bytes: 2_500_000,
                gated: false,
                sha256: None,
            },
        ],
        defaults: TTS_DEFAULTS,
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

static KNOWN_MANIFESTS: LazyLock<Vec<ModelManifest<TTSComponent>>> = LazyLock::new(|| {
    vec![parler_mini_manifest(), parler_large_manifest()]
});

static MANIFEST_INDEX: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    KNOWN_MANIFESTS
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.clone(), i))
        .collect()
});

pub fn known_manifests() -> &'static [ModelManifest<TTSComponent>] {
    &KNOWN_MANIFESTS
}

pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<TTSComponent>> {
    let canonical = resolve_model_name(name);
    MANIFEST_INDEX.get(&canonical).map(|&i| &KNOWN_MANIFESTS[i])
}

/// Resolve shorthand model names to canonical form.
pub fn resolve_model_name(name: &str) -> String {
    // Already in registry
    if MANIFEST_INDEX.contains_key(name) {
        return name.to_string();
    }
    // Try with "parler-" prefix
    let prefixed = format!("parler-{name}");
    if MANIFEST_INDEX.contains_key(&prefixed) {
        return prefixed;
    }
    name.to_string()
}

/// Total download size in bytes for a manifest.
pub fn total_download_size(manifest: &ModelManifest<TTSComponent>) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}
