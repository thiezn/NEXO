use local_inference_helpers::manifest::{Component, ManifestDefaults, ModelFile, ModelManifest};
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMComponent {
    Model,
    ModelShard,
    Tokenizer,
    Config,
    PreprocessorConfig,
}

impl Component for MMComponent {
    fn name(&self) -> &str {
        match self {
            Self::Model => "model",
            Self::ModelShard => "model_shard",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::PreprocessorConfig => "preprocessor_config",
        }
    }

    fn is_model_specific(&self) -> bool {
        matches!(self, Self::Model | Self::ModelShard)
    }
}

const MM_DEFAULTS: ManifestDefaults = ManifestDefaults {
    steps: 0,
    guidance: 0.0,
    width: 0,
    height: 0,
};

fn metadata_files(hf_repo: &str) -> Vec<ModelFile<MMComponent>> {
    vec![
        ModelFile {
            component: MMComponent::Tokenizer,
            hf_repo: hf_repo.to_string(),
            hf_filename: "tokenizer.json".to_string(),
            size_bytes: 11_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: MMComponent::Config,
            hf_repo: hf_repo.to_string(),
            hf_filename: "config.json".to_string(),
            size_bytes: 5_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: MMComponent::PreprocessorConfig,
            hf_repo: hf_repo.to_string(),
            hf_filename: "preprocessor_config.json".to_string(),
            size_bytes: 500,
            gated: false,
            sha256: None,
        },
    ]
}

// Qwen3.5-9B: dense multimodal model, 4 shards, ~19.3 GB total
static QWEN35_9B: LazyLock<ModelManifest<MMComponent>> = LazyLock::new(|| {
    let hf_repo = "Qwen/Qwen3.5-9B";
    let mut files = metadata_files(hf_repo);
    let num_shards = 4;
    let shard_size = 19_306_216_416u64 / num_shards as u64;
    for i in 1..=num_shards {
        files.push(ModelFile {
            component: if i == 1 {
                MMComponent::Model
            } else {
                MMComponent::ModelShard
            },
            hf_repo: hf_repo.to_string(),
            hf_filename: format!("model.safetensors-{i:05}-of-{num_shards:05}.safetensors"),
            size_bytes: shard_size,
            gated: false,
            sha256: None,
        });
    }
    ModelManifest {
        name: "qwen3.5-9b".to_string(),
        family: "qwen3.5".to_string(),
        description: "Qwen3.5 9B -- multimodal (text, image, video)".to_string(),
        size_gb: 19.3,
        files,
        defaults: MM_DEFAULTS,
    }
});

// Qwen3.5-35B-A3B: MoE multimodal model, 14 shards, ~71.9 GB total
static QWEN35_35B_A3B: LazyLock<ModelManifest<MMComponent>> = LazyLock::new(|| {
    let hf_repo = "Qwen/Qwen3.5-35B-A3B";
    let mut files = metadata_files(hf_repo);
    let num_shards = 14;
    let shard_size = 71_903_655_008u64 / num_shards as u64;
    for i in 1..=num_shards {
        files.push(ModelFile {
            component: if i == 1 {
                MMComponent::Model
            } else {
                MMComponent::ModelShard
            },
            hf_repo: hf_repo.to_string(),
            hf_filename: format!("model.safetensors-{i:05}-of-{num_shards:05}.safetensors"),
            size_bytes: shard_size,
            gated: false,
            sha256: None,
        });
    }
    ModelManifest {
        name: "qwen3.5-35b-a3b".to_string(),
        family: "qwen3.5-moe".to_string(),
        description: "Qwen3.5 35B-A3B -- MoE multimodal (text, image, video)".to_string(),
        size_gb: 71.9,
        files,
        defaults: MM_DEFAULTS,
    }
});

pub fn known_manifests() -> Vec<&'static ModelManifest<MMComponent>> {
    vec![&QWEN35_9B, &QWEN35_35B_A3B]
}

pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<MMComponent>> {
    let canonical = resolve_model_name(name);
    known_manifests()
        .into_iter()
        .find(|m| m.name == canonical)
}

pub fn resolve_model_name(name: &str) -> String {
    match name {
        "9b" | "qwen3.5-9b" => "qwen3.5-9b".to_string(),
        "35b" | "qwen3.5-35b" | "qwen3.5-35b-a3b" | "35b-a3b" => {
            "qwen3.5-35b-a3b".to_string()
        }
        other => other.to_string(),
    }
}

pub fn total_download_size(manifest: &ModelManifest<MMComponent>) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}
