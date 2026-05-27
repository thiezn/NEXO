//! Built-in model registry used by the reusable `models` command.

use std::path::Path;
use std::sync::LazyLock;

use nexo_core::{
    MetadataMap, ModelCapability, ModelDescriptor, ModelModalities, RoleStrategy, SupportedModality,
};
use serde_json::Value;

use crate::manifest::{
    ModelComponent, ModelFile, ModelFileSelector, ModelManifest, sanitize_model_name,
};
use crate::paths::default_models_dir;

/// Printable model entry with local download status.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Stable model identifier.
    pub id: String,
    /// Human-readable model label.
    pub display_name: String,
    /// Model provider or publishing organization.
    pub provider: Option<String>,
    /// Model family identifier.
    pub family: String,
    /// Loader/runtime backend label.
    pub backend: String,
    /// Declared model capabilities.
    pub capabilities: Vec<ModelCapability>,
    /// Declared input and output modalities.
    pub modalities: ModelModalities,
    /// Approximate total size in GB.
    pub size_gb: f32,
    /// Human-readable description.
    pub description: String,
    /// Whether the known local artifact selectors are present.
    pub is_downloaded: bool,
}

/// Returns all known manifests.
#[must_use]
pub fn known_manifests() -> &'static [ModelManifest] {
    &ALL_MANIFESTS
}

/// Finds a manifest by model id or display name.
#[must_use]
pub fn find_manifest(name: &str) -> Option<&'static ModelManifest> {
    known_manifests().iter().find(|manifest| {
        manifest.id() == name || manifest.display_name().eq_ignore_ascii_case(name)
    })
}

/// Finds a manifest by model id, display name, or any Hugging Face source repository used by its files.
#[must_use]
pub fn find_manifest_by_source(source: &str) -> Option<&'static ModelManifest> {
    known_manifests().iter().find(|manifest| {
        manifest.id() == source
            || manifest.display_name().eq_ignore_ascii_case(source)
            || manifest.files.iter().any(|file| file.hf_repo == source)
    })
}

/// Returns all manifests that advertise a core model capability.
#[must_use]
pub fn manifests_for_capability(capability: ModelCapability) -> Vec<&'static ModelManifest> {
    known_manifests()
        .iter()
        .filter(|manifest| manifest.descriptor.capabilities.contains(&capability))
        .collect()
}

/// Returns all manifests that accept or emit a core modality.
#[must_use]
pub fn manifests_for_modality(modality: SupportedModality) -> Vec<&'static ModelManifest> {
    known_manifests()
        .iter()
        .filter(|manifest| {
            manifest.descriptor.modalities.input.contains(&modality)
                || manifest.descriptor.modalities.output.contains(&modality)
        })
        .collect()
}

/// Stable lowercase capability label for CLI output.
#[must_use]
pub const fn capability_label(capability: ModelCapability) -> &'static str {
    match capability {
        ModelCapability::TextGeneration => "chat",
        ModelCapability::ToolCalling => "tool",
        ModelCapability::Embeddings => "embedding",
        ModelCapability::ImageInput => "image-in",
        ModelCapability::VideoInput => "video-in",
        ModelCapability::AudioInput => "audio-in",
        ModelCapability::ImageGeneration => "image-gen",
        ModelCapability::SpeechGeneration => "speech-gen",
        ModelCapability::StructuredOutput => "structured",
        ModelCapability::Reasoning => "reasoning",
        ModelCapability::Streaming => "streaming",
    }
}

/// Build a printable list of all known models.
#[must_use]
pub fn list_models() -> Vec<ModelEntry> {
    let models_dir = default_models_dir();

    known_manifests()
        .iter()
        .map(|manifest| {
            let model_dir = models_dir.join(sanitize_model_name(manifest.id()));
            ModelEntry {
                id: manifest.id().to_string(),
                display_name: manifest.display_name().to_string(),
                provider: manifest.descriptor.provider.clone(),
                family: metadata_string(&manifest.descriptor.metadata, "family"),
                backend: manifest.backend.clone(),
                capabilities: manifest.descriptor.capabilities.clone(),
                modalities: manifest.descriptor.modalities.clone(),
                size_gb: manifest.size_gb,
                description: metadata_string(&manifest.descriptor.metadata, "description"),
                is_downloaded: !manifest.files.is_empty()
                    && manifest
                        .files
                        .iter()
                        .all(|file| local_selector_present(&model_dir, &file.selector)),
            }
        })
        .collect()
}

static ALL_MANIFESTS: LazyLock<Vec<ModelManifest>> = LazyLock::new(|| {
    let mut manifests = vec![
        gemma_4_e2b_it_q5_manifest(),
        gemma_4_26b_a4b_it_q4_manifest(),
    ];

    manifests.extend(gemma_4_uqff_manifests());
    manifests.extend(qwen_3_5_apple_metal_manifests());
    manifests.push(voxtral_mini_asr_stt_manifest());
    manifests.extend(flux_2_manifests());
    manifests.push(embedding_gemma_manifest());
    manifests
});

fn gemma_4_e2b_it_q5_manifest() -> ModelManifest {
    let gguf_repo = "unsloth/gemma-4-e2b-it-GGUF";
    let orig_repo = "google/gemma-4-E2B-it";
    ModelManifest {
        descriptor: descriptor(
            "gemma-4-e2b-it-q5",
            "Gemma 4 E2B IT Q5_K_M",
            Some("unsloth"),
            "gemma4",
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
                ModelCapability::ImageInput,
                ModelCapability::VideoInput,
                ModelCapability::AudioInput,
                ModelCapability::StructuredOutput,
                ModelCapability::Streaming,
            ],
            vec![
                SupportedModality::Text,
                SupportedModality::Image,
                SupportedModality::Video,
                SupportedModality::Audio,
            ],
            vec![SupportedModality::Text],
            "Gemma 4 E2B instruction-tuned GGUF multimodal chat model.",
            orig_repo,
            "gguf",
            None,
        ),
        backend: "mistralrs-gguf".to_string(),
        size_gb: 4.1,
        files: vec![
            file_exact(
                ModelComponent::Weights,
                gguf_repo,
                "gemma-4-E2B-it-Q5_K_M.gguf",
                Some(3_356_030_336),
                false,
            ),
            file_exact(
                ModelComponent::VisionProjector,
                gguf_repo,
                "mmproj-F16.gguf",
                Some(985_654_208),
                false,
            ),
            file_exact(
                ModelComponent::Tokenizer,
                orig_repo,
                "tokenizer.json",
                Some(32_169_626),
                true,
            ),
            file_exact(
                ModelComponent::Config,
                orig_repo,
                "config.json",
                Some(4_954),
                true,
            ),
        ],
    }
}

fn gemma_4_26b_a4b_it_q4_manifest() -> ModelManifest {
    let gguf_repo = "unsloth/gemma-4-26b-a4b-it-GGUF";
    let orig_repo = "google/gemma-4-26b-a4b-it";
    ModelManifest {
        descriptor: descriptor(
            "gemma-4-26b-a4b-it-q4",
            "Gemma 4 26B-A4B IT UD-Q4_K_M",
            Some("unsloth"),
            "gemma4",
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
                ModelCapability::ImageInput,
                ModelCapability::StructuredOutput,
                ModelCapability::Streaming,
            ],
            vec![SupportedModality::Text, SupportedModality::Image],
            vec![SupportedModality::Text],
            "Gemma 4 26B-A4B instruction-tuned GGUF multimodal MoE model.",
            orig_repo,
            "gguf",
            None,
        ),
        backend: "mistralrs-gguf".to_string(),
        size_gb: 17.0,
        files: vec![
            file_exact(
                ModelComponent::Weights,
                gguf_repo,
                "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                Some(16_868_236_288),
                false,
            ),
            file_exact(
                ModelComponent::VisionProjector,
                gguf_repo,
                "mmproj-F16.gguf",
                Some(1_193_058_912),
                false,
            ),
            file_exact(
                ModelComponent::Tokenizer,
                orig_repo,
                "tokenizer.json",
                Some(32_169_626),
                true,
            ),
            file_exact(
                ModelComponent::Config,
                orig_repo,
                "config.json",
                Some(4_954),
                true,
            ),
        ],
    }
}

fn gemma_4_uqff_manifests() -> Vec<ModelManifest> {
    [
        (
            "gemma-4-e2b-uqff",
            "Gemma 4 E2B UQFF",
            "google/gemma-4-E2B",
            4.0,
        ),
        (
            "gemma-4-e2b-it-uqff",
            "Gemma 4 E2B IT UQFF",
            "google/gemma-4-E2B-it",
            4.0,
        ),
        (
            "gemma-4-e4b-uqff",
            "Gemma 4 E4B UQFF",
            "google/gemma-4-E4B",
            7.0,
        ),
        (
            "gemma-4-e4b-it-uqff",
            "Gemma 4 E4B IT UQFF",
            "google/gemma-4-E4B-it",
            7.0,
        ),
    ]
    .into_iter()
    .map(|(id, display_name, source_model, size_gb)| {
        let base_name = source_model
            .rsplit_once('/')
            .map_or(source_model, |(_, name)| name);
        let uqff_repo = format!("mistralrs-community/{base_name}-UQFF");
        ModelManifest {
            descriptor: descriptor(
                id,
                display_name,
                Some("mistralrs-community"),
                "gemma4",
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                    ModelCapability::VideoInput,
                    ModelCapability::AudioInput,
                    ModelCapability::StructuredOutput,
                    ModelCapability::Streaming,
                ],
                vec![
                    SupportedModality::Text,
                    SupportedModality::Image,
                    SupportedModality::Video,
                    SupportedModality::Audio,
                ],
                vec![SupportedModality::Text],
                "Mistral UQFF Gemma 4 artifact set with quantized shards and residual tensors.",
                source_model,
                "uqff",
                None,
            ),
            backend: "mistralrs-uqff".to_string(),
            size_gb,
            files: files_for_uqff_repo(&uqff_repo, true),
        }
    })
    .collect()
}

fn qwen_3_5_apple_metal_manifests() -> Vec<ModelManifest> {
    [
        (
            "qwen3.5-27b-apple-metal-uqff",
            "Qwen 3.5 27B Apple Metal UQFF",
            "Qwen/Qwen3.5-27B",
            18.0,
        ),
        (
            "qwen3.5-35b-a3b-apple-metal-uqff",
            "Qwen 3.5 35B-A3B Apple Metal UQFF",
            "Qwen/Qwen3.5-35B-A3B",
            22.0,
        ),
    ]
    .into_iter()
    .map(|(id, display_name, source_model, size_gb)| {
        let base_name = source_model
            .rsplit_once('/')
            .map_or(source_model, |(_, name)| name);
        let uqff_repo = format!("mistralrs-community/{base_name}-UQFF");
        let mut manifest = ModelManifest {
            descriptor: descriptor(
                id,
                display_name,
                Some("mistralrs-community"),
                "qwen3.5",
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                    ModelCapability::Streaming,
                ],
                vec![SupportedModality::Text, SupportedModality::Image],
                vec![SupportedModality::Text],
                "Mistral UQFF Qwen 3.5 artifact set intended for Apple Metal AFQ execution.",
                source_model,
                "uqff",
                None,
            ),
            backend: "mistralrs-uqff-metal".to_string(),
            size_gb,
            files: files_for_uqff_repo(&uqff_repo, false),
        };
        manifest.descriptor.metadata.insert(
            "optimization".to_string(),
            Value::String("apple-metal-afq".to_string()),
        );
        manifest
    })
    .collect()
}

fn voxtral_mini_asr_stt_manifest() -> ModelManifest {
    let repo = "mistralai/Voxtral-Mini-3B-2507";
    ModelManifest {
        descriptor: descriptor(
            "voxtral-mini-3b-2507-asr-stt",
            "Voxtral Mini 3B 2507 ASR/STT",
            Some("mistralai"),
            "voxtral",
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::AudioInput,
                ModelCapability::Streaming,
            ],
            vec![SupportedModality::Text, SupportedModality::Audio],
            vec![SupportedModality::Text],
            "Voxtral speech-to-text and audio understanding model for ASR/STT workflows.",
            repo,
            "safetensors",
            None,
        ),
        backend: "mistralrs-voxtral".to_string(),
        size_gb: 7.0,
        files: files_for_hf_snapshot(repo, false),
    }
}

fn flux_2_manifests() -> Vec<ModelManifest> {
    [
        (
            "flux.2-dev",
            "FLUX.2 Dev",
            "black-forest-labs/FLUX.2-dev",
            24.0,
        ),
        (
            "flux.2-schnell",
            "FLUX.2 Schnell",
            "black-forest-labs/FLUX.2-schnell",
            24.0,
        ),
    ]
    .into_iter()
    .map(|(id, display_name, repo, size_gb)| ModelManifest {
        descriptor: descriptor(
            id,
            display_name,
            Some("black-forest-labs"),
            "flux2",
            vec![ModelCapability::ImageGeneration],
            vec![SupportedModality::Text],
            vec![SupportedModality::Image],
            "FLUX.2 image generation artifact snapshot.",
            repo,
            "diffusers",
            None,
        ),
        backend: "mistralrs-flux".to_string(),
        size_gb,
        files: files_for_hf_snapshot(repo, true),
    })
    .collect()
}

fn embedding_gemma_manifest() -> ModelManifest {
    let repo = "google/embeddinggemma-300m";
    ModelManifest {
        descriptor: descriptor(
            "embedding-gemma-300m",
            "Embedding Gemma 300M",
            Some("google"),
            "embedding-gemma",
            vec![ModelCapability::Embeddings],
            vec![SupportedModality::Text],
            Vec::new(),
            "Google Embedding Gemma model for text embedding vectors.",
            repo,
            "safetensors",
            Some(2048),
        ),
        backend: "mistralrs-embedding".to_string(),
        size_gb: 1.0,
        files: files_for_hf_snapshot(repo, true),
    }
}

#[allow(clippy::too_many_arguments)]
fn descriptor(
    id: &str,
    display_name: &str,
    provider: Option<&str>,
    family: &str,
    capabilities: Vec<ModelCapability>,
    input_modalities: Vec<SupportedModality>,
    output_modalities: Vec<SupportedModality>,
    description: &str,
    source_model: &str,
    artifact_format: &str,
    context_window_tokens: Option<usize>,
) -> ModelDescriptor {
    let mut metadata = MetadataMap::new();
    metadata.insert("family".to_string(), Value::String(family.to_string()));
    metadata.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    metadata.insert(
        "source_model".to_string(),
        Value::String(source_model.to_string()),
    );
    metadata.insert(
        "artifact_format".to_string(),
        Value::String(artifact_format.to_string()),
    );

    ModelDescriptor {
        id: id.into(),
        display_name: display_name.to_string(),
        provider: provider.map(str::to_string),
        capabilities,
        modalities: ModelModalities {
            input: input_modalities,
            output: output_modalities,
        },
        role_strategy: RoleStrategy::Default,
        context_window_tokens,
        max_output_tokens: None,
        metadata,
    }
}

fn file_exact(
    component: ModelComponent,
    hf_repo: &str,
    filename: &str,
    size_bytes: Option<u64>,
    gated: bool,
) -> ModelFile {
    ModelFile {
        component,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Exact(filename.to_string()),
        size_bytes,
        gated,
        sha256: None,
    }
}

fn file_suffix(component: ModelComponent, hf_repo: &str, suffix: &str, gated: bool) -> ModelFile {
    ModelFile {
        component,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Suffix(suffix.to_string()),
        size_bytes: None,
        gated,
        sha256: None,
    }
}

fn files_for_uqff_repo(hf_repo: &str, gated: bool) -> Vec<ModelFile> {
    vec![
        file_suffix(ModelComponent::UqffShard, hf_repo, ".uqff", gated),
        file_suffix(ModelComponent::UqffResidual, hf_repo, ".safetensors", gated),
        file_suffix(ModelComponent::Config, hf_repo, ".json", gated),
    ]
}

fn files_for_hf_snapshot(hf_repo: &str, gated: bool) -> Vec<ModelFile> {
    vec![
        file_suffix(ModelComponent::WeightShard, hf_repo, ".safetensors", gated),
        file_suffix(ModelComponent::Config, hf_repo, ".json", gated),
    ]
}

fn metadata_string(metadata: &MetadataMap, key: &str) -> String {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn local_selector_present(model_dir: &Path, selector: &ModelFileSelector) -> bool {
    match selector {
        ModelFileSelector::Exact(path) => model_dir.join(path).exists(),
        ModelFileSelector::Suffix(_) | ModelFileSelector::Prefix(_) => {
            any_local_file_matches(model_dir, selector)
        }
    }
}

fn any_local_file_matches(model_dir: &Path, selector: &ModelFileSelector) -> bool {
    let mut stack = vec![model_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }

            if let Ok(relative) = path.strip_prefix(model_dir)
                && selector.matches(&relative.to_string_lossy())
            {
                return true;
            }
        }
    }

    false
}
