//! Built-in model registry used by the reusable `models` command.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use nexo_core::{
    InferenceRuntime, MetadataMap, ModelCapability, ModelDescriptor, ModelModalities,
    RoleStrategy, SupportedModality,
};
use serde_json::Value;

use crate::manifest::{ModelComponent, ModelFile, ModelFileSelector, ModelManifest};
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
            let model_dir = models_dir.join(manifest.storage_id());
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
                        .all(|file| local_file_present(&model_dir, file))
                    && local_safetensors_indexes_complete(&model_dir),
            }
        })
        .collect()
}

static ALL_MANIFESTS: LazyLock<Vec<ModelManifest>> = LazyLock::new(|| {
    let mut manifests = vec![
        gemma_4_e2b_it_q5_manifest(),
        gemma_4_12b_it_manifest(),
        gemma_4_26b_a4b_it_q4_manifest(),
    ];

    manifests.extend(gemma_4_uqff_manifests());
    manifests.extend(qwen_3_5_apple_metal_manifests());
    manifests.push(voxtral_mini_asr_stt_manifest());
    manifests.push(dia_1_6b_tts_manifest());
    manifests.push(kokoro_82m_tts_manifest());
    manifests.extend(flux_2_manifests());
    manifests.push(embedding_gemma_manifest());
    manifests
});

const UQFF_VARIANTS: &[(&str, &str)] = &[
    ("afq2", "AFQ2"),
    ("afq3", "AFQ3"),
    ("afq4", "AFQ4"),
    ("afq6", "AFQ6"),
    ("afq8", "AFQ8"),
    ("q2k", "Q2K"),
    ("q3k", "Q3K"),
    ("q4k", "Q4K"),
    ("q5k", "Q5K"),
    ("q6k", "Q6K"),
    ("q8_0", "Q8_0"),
];

fn gemma_4_e2b_it_q5_manifest() -> ModelManifest {
    let gguf_repo = "unsloth/gemma-4-e2b-it-GGUF";
    let orig_repo = "google/gemma-4-E2B-it";
    ModelManifest {
        descriptor: descriptor(
            "gemma-4-e2b-it-q5",
            "Gemma 4 E2B IT Q5_K_M",
            Some("unsloth"),
            InferenceRuntime::MistralRs,
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
            file_chat_template_jinja(orig_repo, true),
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
            InferenceRuntime::MistralRs,
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
            file_chat_template_jinja(orig_repo, true),
        ],
    }
}

fn gemma_4_12b_it_manifest() -> ModelManifest {
    let repo = "google/gemma-4-12B-it";
    let mut files = files_for_hf_snapshot(repo, true);
    files.push(file_chat_template_jinja(repo, true));

    ModelManifest {
        descriptor: descriptor(
            "gemma-4-12b-it",
            "Gemma 4 12B IT",
            Some("google"),
            InferenceRuntime::MistralRs,
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
            "Gemma 4 12B instruction-tuned safetensors multimodal chat model.",
            repo,
            "safetensors",
            None,
        ),
        backend: "mistralrs-gemma4".to_string(),
        size_gb: 26.0,
        files,
    }
}

fn gemma_4_uqff_manifests() -> Vec<ModelManifest> {
    [
        (
            "gemma-4-e2b-it-uqff",
            "Gemma 4 E2B IT UQFF",
            "google/gemma-4-E2B-it",
            4.0,
        ),
        (
            "gemma-4-e4b-it-uqff",
            "Gemma 4 E4B IT UQFF",
            "google/gemma-4-E4B-it",
            7.0,
        ),
        (
            "gemma-4-12b-it-uqff",
            "Gemma 4 12B IT UQFF",
            "google/gemma-4-12B-it",
            12.0,
        ),
        (
            "gemma-4-26b-a4b-it-uqff",
            "Gemma 4 26B-A4B IT UQFF",
            "google/gemma-4-26B-A4B-it",
            17.0,
        ),
        (
            "gemma-4-31b-it-uqff",
            "Gemma 4 31B IT UQFF",
            "google/gemma-4-31B-it",
            20.0,
        ),
    ]
    .into_iter()
    .flat_map(|(id, display_name, source_model, size_gb)| {
        let base_name = source_model
            .rsplit_once('/')
            .map_or(source_model, |(_, name)| name);
        let uqff_repo = format!("mistralrs-community/{base_name}-UQFF");

        UQFF_VARIANTS.iter().map(move |(variant_id, variant_label)| {
            let mut manifest = uqff_variant_manifest(UqffManifestSpec {
                id_prefix: id,
                display_name_prefix: display_name,
                provider: "mistralrs-community",
                family: "gemma4",
                capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                    ModelCapability::VideoInput,
                    ModelCapability::AudioInput,
                    ModelCapability::StructuredOutput,
                    ModelCapability::Streaming,
                ],
                input_modalities: vec![
                    SupportedModality::Text,
                    SupportedModality::Image,
                    SupportedModality::Video,
                    SupportedModality::Audio,
                ],
                output_modalities: vec![SupportedModality::Text],
                description: "Mistral UQFF Gemma 4 artifact set with quantized shards and residual tensors.",
                source_model,
                backend: "mistralrs-uqff",
                size_gb,
                hf_repo: &uqff_repo,
                variant_id,
                variant_label,
                gated: true,
            });
            manifest
                .files
                .push(file_chat_template_jinja(source_model, true));
            manifest
        })
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
    .flat_map(|(id, display_name, source_model, size_gb)| {
        let base_name = source_model
            .rsplit_once('/')
            .map_or(source_model, |(_, name)| name);
        let uqff_repo = format!("mistralrs-community/{base_name}-UQFF");

        UQFF_VARIANTS.iter().map(move |(variant_id, variant_label)| {
            let mut manifest = uqff_variant_manifest(UqffManifestSpec {
                id_prefix: id,
                display_name_prefix: display_name,
                provider: "mistralrs-community",
                family: "qwen3.5",
                capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                    ModelCapability::Streaming,
                ],
                input_modalities: vec![SupportedModality::Text, SupportedModality::Image],
                output_modalities: vec![SupportedModality::Text],
                description: "Mistral UQFF Qwen 3.5 artifact set intended for Apple Metal AFQ execution.",
                source_model,
                backend: "mistralrs-uqff-metal",
                size_gb,
                hf_repo: &uqff_repo,
                variant_id,
                variant_label,
                gated: false,
            });
            manifest.descriptor.metadata.insert(
                "optimization".to_string(),
                Value::String("apple-metal-afq".to_string()),
            );
            manifest
                .files
                .push(file_chat_template_tokenizer_config(&uqff_repo, false));
            manifest
        })
    })
    .collect()
}

struct UqffManifestSpec<'a> {
    id_prefix: &'a str,
    display_name_prefix: &'a str,
    provider: &'a str,
    family: &'a str,
    capabilities: Vec<ModelCapability>,
    input_modalities: Vec<SupportedModality>,
    output_modalities: Vec<SupportedModality>,
    description: &'a str,
    source_model: &'a str,
    backend: &'a str,
    size_gb: f32,
    hf_repo: &'a str,
    variant_id: &'a str,
    variant_label: &'a str,
    gated: bool,
}

fn uqff_variant_manifest(spec: UqffManifestSpec<'_>) -> ModelManifest {
    ModelManifest {
        descriptor: descriptor(
            &format!("{}-{}", spec.id_prefix, spec.variant_id),
            &format!("{} {}", spec.display_name_prefix, spec.variant_label),
            Some(spec.provider),
            InferenceRuntime::MistralRs,
            spec.family,
            spec.capabilities,
            spec.input_modalities,
            spec.output_modalities,
            spec.description,
            spec.source_model,
            "uqff",
            None,
        ),
        backend: spec.backend.to_string(),
        size_gb: spec.size_gb,
        files: files_for_uqff_variant(spec.hf_repo, spec.variant_id, spec.gated),
    }
}

fn voxtral_mini_asr_stt_manifest() -> ModelManifest {
    let repo = "mistralai/Voxtral-Mini-3B-2507";
    ModelManifest {
        descriptor: descriptor(
            "voxtral-mini-3b-2507-asr-stt",
            "Voxtral Mini 3B 2507 ASR/STT",
            Some("mistralai"),
            InferenceRuntime::MistralRs,
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
        files: files_for_chat_snapshot(repo, false),
    }
}

fn dia_1_6b_tts_manifest() -> ModelManifest {
    let repo = "nari-labs/Dia-1.6B";
    let dac_repo = "EricB/dac_44khz";
    ModelManifest {
        descriptor: descriptor(
            "dia-1.6b-tts",
            "Dia 1.6B TTS",
            Some("nari-labs"),
            InferenceRuntime::MistralRs,
            "dia",
            vec![ModelCapability::SpeechGeneration],
            vec![SupportedModality::Text],
            vec![SupportedModality::Audio],
            "Dia text-to-speech model for synthesizing audio from text prompts.",
            repo,
            "safetensors",
            None,
        ),
        backend: "mistralrs-dia".to_string(),
        size_gb: 3.5,
        files: vec![
            file_exact(ModelComponent::Config, repo, "config.json", None, false),
            file_exact(
                ModelComponent::Weights,
                repo,
                "model.safetensors",
                None,
                false,
            ),
            file_exact_at(
                ModelComponent::Weights,
                dac_repo,
                "model.safetensors",
                "dac/model.safetensors",
                None,
                false,
            ),
        ],
    }
}

fn kokoro_82m_tts_manifest() -> ModelManifest {
    let repo = "hexgrad/Kokoro-82M";
    ModelManifest {
        descriptor: descriptor(
            "kokoro-82m-tts",
            "Kokoro 82M TTS",
            Some("hexgrad"),
            InferenceRuntime::Any,
            "kokoro",
            vec![ModelCapability::SpeechGeneration],
            vec![SupportedModality::Text],
            vec![SupportedModality::Audio],
            "Kokoro text-to-speech model for fast local speech synthesis.",
            repo,
            "pytorch",
            None,
        ),
        backend: "any-tts-kokoro".to_string(),
        size_gb: 0.4,
        files: vec![
            file_exact(ModelComponent::Config, repo, "config.json", None, false),
            file_exact(
                ModelComponent::Weights,
                repo,
                "kokoro-v1_0.pth",
                None,
                false,
            ),
            file_prefix(ModelComponent::Modules, repo, "voices/", false),
        ],
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
        (
            "flux.2-klein-4b",
            "FLUX.2 Klein 4B",
            "black-forest-labs/FLUX.2-klein-4B",
            13.0,
        ),
        (
            "flux.2-klein-9b",
            "FLUX.2 Klein 9B",
            "black-forest-labs/FLUX.2-klein-9B",
            29.0,
        ),
    ]
    .into_iter()
    .map(|(id, display_name, repo, size_gb)| ModelManifest {
        descriptor: descriptor(
            id,
            display_name,
            Some("black-forest-labs"),
            InferenceRuntime::Mold,
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
            InferenceRuntime::MistralRs,
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
    runtime: InferenceRuntime,
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
        runtime,
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
        local_path: None,
        size_bytes,
        gated,
        sha256: None,
    }
}

fn file_exact_at(
    component: ModelComponent,
    hf_repo: &str,
    filename: &str,
    local_path: &str,
    size_bytes: Option<u64>,
    gated: bool,
) -> ModelFile {
    ModelFile {
        component,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Exact(filename.to_string()),
        local_path: Some(local_path.to_string()),
        size_bytes,
        gated,
        sha256: None,
    }
}

fn file_chat_template_exact(hf_repo: &str, filename: &str, gated: bool) -> ModelFile {
    ModelFile {
        component: ModelComponent::ChatTemplate,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Exact(filename.to_string()),
        local_path: None,
        size_bytes: None,
        gated,
        sha256: None,
    }
}

fn file_chat_template_jinja(hf_repo: &str, gated: bool) -> ModelFile {
    file_chat_template_exact(hf_repo, "chat_template.jinja", gated)
}

fn file_chat_template_tokenizer_config(hf_repo: &str, gated: bool) -> ModelFile {
    file_chat_template_exact(hf_repo, "tokenizer_config.json", gated)
}

fn file_suffix(component: ModelComponent, hf_repo: &str, suffix: &str, gated: bool) -> ModelFile {
    ModelFile {
        component,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Suffix(suffix.to_string()),
        local_path: None,
        size_bytes: None,
        gated,
        sha256: None,
    }
}

fn file_prefix(component: ModelComponent, hf_repo: &str, prefix: &str, gated: bool) -> ModelFile {
    ModelFile {
        component,
        hf_repo: hf_repo.to_string(),
        selector: ModelFileSelector::Prefix(prefix.to_string()),
        local_path: None,
        size_bytes: None,
        gated,
        sha256: None,
    }
}

fn files_for_uqff_variant(hf_repo: &str, variant: &str, gated: bool) -> Vec<ModelFile> {
    vec![
        file_prefix(
            ModelComponent::UqffShard,
            hf_repo,
            &format!("{variant}-"),
            gated,
        ),
        file_suffix(ModelComponent::UqffResidual, hf_repo, ".safetensors", gated),
        file_suffix(ModelComponent::Config, hf_repo, ".json", gated),
    ]
}

fn files_for_chat_snapshot(hf_repo: &str, gated: bool) -> Vec<ModelFile> {
    let mut files = files_for_hf_snapshot(hf_repo, gated);
    files.push(file_chat_template_tokenizer_config(hf_repo, gated));
    files
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

fn local_file_present(model_dir: &Path, file: &ModelFile) -> bool {
    if let Some(local_path) = &file.local_path {
        return model_dir.join(local_path).exists();
    }

    match &file.selector {
        ModelFileSelector::Exact(path) => model_dir.join(path).exists(),
        ModelFileSelector::Suffix(_) | ModelFileSelector::Prefix(_) => {
            any_local_file_matches(model_dir, &file.selector)
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

fn local_safetensors_indexes_complete(model_dir: &Path) -> bool {
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

            if path
                .file_name()
                .and_then(|filename| filename.to_str())
                .is_some_and(|filename| filename.ends_with(".safetensors.index.json"))
                && !safetensors_index_shards_present(&path)
            {
                tracing::warn!(
                    index = %path.display(),
                    "Safetensors index references missing local shards"
                );
                return false;
            }
        }
    }

    true
}

fn safetensors_index_shards_present(index_path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(index_path) else {
        return false;
    };
    let Ok(value) = serde_json::from_reader::<_, Value>(file) else {
        return false;
    };
    let Some(weight_map) = value.get("weight_map").and_then(Value::as_object) else {
        return true;
    };

    let Some(parent) = index_path.parent() else {
        return false;
    };

    let shard_paths = weight_map
        .values()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();

    shard_paths
        .into_iter()
        .all(|shard_path| parent.join(shard_path).exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_models_include_chat_template_component() {
        let chat_manifest_ids = [
            "gemma-4-e2b-it-q5",
            "gemma-4-12b-it",
            "gemma-4-26b-a4b-it-q4",
            "gemma-4-e2b-it-uqff-afq4",
            "gemma-4-12b-it-uqff-afq4",
            "qwen3.5-27b-apple-metal-uqff-afq4",
            "voxtral-mini-3b-2507-asr-stt",
        ];

        for id in chat_manifest_ids {
            let manifest = find_manifest(id).unwrap_or_else(|| panic!("missing manifest {id}"));
            assert!(
                manifest
                    .files
                    .iter()
                    .any(|file| file.component == ModelComponent::ChatTemplate),
                "manifest {id} is missing a chat template component"
            );
        }
    }

    #[test]
    fn gemma4_models_use_jinja_chat_template() {
        let gemma_manifest_ids = [
            "gemma-4-e2b-it-q5",
            "gemma-4-12b-it",
            "gemma-4-26b-a4b-it-q4",
            "gemma-4-e2b-it-uqff-afq4",
            "gemma-4-e4b-it-uqff-afq4",
            "gemma-4-12b-it-uqff-afq4",
            "gemma-4-26b-a4b-it-uqff-afq4",
            "gemma-4-31b-it-uqff-afq4",
        ];

        for id in gemma_manifest_ids {
            let manifest = find_manifest(id).unwrap_or_else(|| panic!("missing manifest {id}"));
            assert!(
                manifest.files.iter().any(|file| {
                    file.component == ModelComponent::ChatTemplate
                        && file.selector.exact_path() == Some("chat_template.jinja")
                }),
                "manifest {id} is missing chat_template.jinja"
            );
        }
    }

    #[test]
    fn gemma_4_12b_manifests_are_cataloged() {
        let manifest = find_manifest("gemma-4-12b-it").expect("missing Gemma 4 12B manifest");
        assert_eq!(manifest.backend, "mistralrs-gemma4");
        assert!(manifest.files.iter().any(|file| {
            file.component == ModelComponent::Config
                && matches!(file.selector, ModelFileSelector::Suffix(ref suffix) if suffix == ".json")
        }));
        assert!(manifest.files.iter().any(|file| {
            file.component == ModelComponent::WeightShard
                && matches!(file.selector, ModelFileSelector::Suffix(ref suffix) if suffix == ".safetensors")
        }));

        let uqff =
            find_manifest("gemma-4-12b-it-uqff-afq8").expect("missing Gemma 4 12B UQFF manifest");
        assert_eq!(uqff.backend, "mistralrs-uqff");
        assert!(uqff.files.iter().any(|file| {
            file.component == ModelComponent::UqffShard
                && matches!(file.selector, ModelFileSelector::Prefix(ref prefix) if prefix == "afq8-")
        }));
    }

    #[test]
    fn dia_manifest_supports_speech_generation_and_dac_sidecar() {
        let manifest = find_manifest("dia-1.6b-tts").expect("missing Dia manifest");

        assert_eq!(manifest.backend, "mistralrs-dia");
        assert!(
            manifest
                .descriptor
                .capabilities
                .contains(&ModelCapability::SpeechGeneration)
        );
        assert_eq!(
            manifest.descriptor.modalities.input,
            vec![SupportedModality::Text]
        );
        assert_eq!(
            manifest.descriptor.modalities.output,
            vec![SupportedModality::Audio]
        );
        assert!(manifest.files.iter().any(|file| {
            file.hf_repo == "EricB/dac_44khz"
                && file.selector.exact_path() == Some("model.safetensors")
                && file.local_path.as_deref() == Some("dac/model.safetensors")
        }));
    }

    #[test]
    fn kokoro_manifest_supports_speech_generation_and_voice_assets() {
        let manifest = find_manifest("kokoro-82m-tts").expect("missing Kokoro manifest");

        assert_eq!(manifest.backend, "any-tts-kokoro");
        assert!(
            manifest
                .descriptor
                .capabilities
                .contains(&ModelCapability::SpeechGeneration)
        );
        assert!(manifest.files.iter().any(|file| {
            file.component == ModelComponent::Weights
                && file.selector.exact_path() == Some("kokoro-v1_0.pth")
        }));
        assert!(manifest.files.iter().any(|file| {
            file.component == ModelComponent::Modules
                && matches!(file.selector, ModelFileSelector::Prefix(ref prefix) if prefix == "voices/")
        }));
    }
}
