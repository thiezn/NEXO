use std::sync::LazyLock;

use crate::download::manifest::{GgufComponent, ModelFile, ModelManifest};

pub type GgufManifest = ModelManifest<GgufComponent>;

/// The primary inference model used by nexo-node.
pub const DEFAULT_INFERENCE_MODEL: &str = "qwen3.5-35b-ab3b";

static ALL_MANIFESTS: LazyLock<Vec<GgufManifest>> = LazyLock::new(|| {
    vec![ModelManifest {
        name: DEFAULT_INFERENCE_MODEL.to_string(),
        family: "qwen3.5".to_string(),
        description: "Qwen3.5 35B-AB3B Q4_K_M GGUF for llama-server (~20 GB)".to_string(),
        size_gb: 20.0,
        files: vec![ModelFile {
            component: GgufComponent::Weights,
            hf_repo: "unsloth/Qwen3.5-35B-A3B-GGUF".to_string(),
            hf_filename: "Qwen3.5-35B-A3B-Q4_K_M.gguf".to_string(),
            // Set to actual byte count after verifying on HuggingFace.
            // 0 disables size-based skip; SHA-based or force-flag checks still apply.
            size_bytes: 0,
            gated: false,
            // Set to upstream SHA-256 once verified.
            sha256: None,
        }],
    }]
});

pub fn known_manifests() -> &'static [GgufManifest] {
    &ALL_MANIFESTS
}

pub fn find_manifest(name: &str) -> Option<&'static GgufManifest> {
    ALL_MANIFESTS.iter().find(|m| m.name == name)
}
