use std::sync::LazyLock;

use crate::download::manifest::{GgufComponent, ModelFile, ModelManifest};

pub type GgufManifest = ModelManifest<GgufComponent>;

/// The primary inference model used by nexo-node.
pub const DEFAULT_INFERENCE_MODEL: &str = "qwen3.5-35b-ab3b";

static ALL_MANIFESTS: LazyLock<Vec<GgufManifest>> = LazyLock::new(|| {
    vec![ModelManifest {
        name: DEFAULT_INFERENCE_MODEL.to_string(),
        family: "qwen3.5".to_string(),
        description: "Qwen3.5 35B-A3B Q4_K_M GGUF + vision projector for llama-server (~21 GB)".to_string(),
        size_gb: 21.0,
        files: vec![
            ModelFile {
                component: GgufComponent::Weights,
                hf_repo: "unsloth/Qwen3.5-35B-A3B-GGUF".to_string(),
                hf_filename: "Qwen3.5-35B-A3B-Q4_K_M.gguf".to_string(),
                size_bytes: 0,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: GgufComponent::VisionProjector,
                hf_repo: "unsloth/Qwen3.5-35B-A3B-GGUF".to_string(),
                hf_filename: "mmproj-F16.gguf".to_string(),
                size_bytes: 0,
                gated: false,
                sha256: None,
            },
        ],
    }]
});

pub fn known_manifests() -> &'static [GgufManifest] {
    &ALL_MANIFESTS
}

pub fn find_manifest(name: &str) -> Option<&'static GgufManifest> {
    ALL_MANIFESTS.iter().find(|m| m.name == name)
}
