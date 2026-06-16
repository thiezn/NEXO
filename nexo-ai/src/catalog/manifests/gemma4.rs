use super::super::{ModelFile, ModelFileKind};
use crate::ModelManifest;
use nexo_core::ModelId;
use std::sync::LazyLock;

// Gemma 4 E4B-it models with UQFF.
const GEMMA_4_E4B_IT_UQFF_REPO: &str = "mistralrs-community/gemma-4-e4b-it-UQFF";

/// Gemma 4 E4B-it model with UQFF.
pub static GEMMA_4_E4B_IT_UQFF_Q8_0: LazyLock<ModelManifest> = LazyLock::new(|| {
    ModelManifest::new_with_folder(
        ModelId::Gemma4E4bItUqffQ80,
        4.0,
        vec![
            ModelFile::new(
                ModelFileKind::ChatTemplate,
                "google/gemma-4-E4B-it",
                "chat-template.jinja",
                10_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
            ModelFile::new_with_suffix(
                ModelFileKind::Config,
                GEMMA_4_E4B_IT_UQFF_REPO,
                ".json",
                50_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
            ModelFile::new_with_suffix(
                ModelFileKind::UqffResidual,
                GEMMA_4_E4B_IT_UQFF_REPO,
                ".safetensors",
                50_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
            ModelFile::new_with_prefix(
                ModelFileKind::UqffShard,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "afq8-",
                1_000_000_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
        ],
        "gemma-4-e4b-it",
    )
});

// Gemma 4 26B-26B_A4B it models with UQFF.
const GEMMA_4_26B_A4B_IT_UQFF_REPO: &str = "mistralrs-community/gemma-4-26b-a4b-it-UQFF";

/// Gemma 4 26B A4B-it model with UQFF.
pub static GEMMA_4_26B_A4B_IT_UQFF_Q8_0: LazyLock<ModelManifest> = LazyLock::new(|| {
    ModelManifest::new_with_folder(
        ModelId::Gemma426bA4bItUqffQ80,
        26.0,
        vec![
            ModelFile::new(
                ModelFileKind::ChatTemplate,
                "google/gemma-4-26B-a4B-it",
                "chat-template.jinja",
                10_000, // Placeholder size in bytes
                "placeholder_sha256_chat_template",
            ),
            ModelFile::new_with_suffix(
                ModelFileKind::Config,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                ".json",
                50_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
            ModelFile::new_with_suffix(
                ModelFileKind::UqffResidual,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                ".safetensors",
                50_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
            ModelFile::new_with_prefix(
                ModelFileKind::UqffShard,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "afq8-",
                1_000_000_000, // Placeholder size in bytes
                "placeholder_sha256",
            ),
        ],
        "gemma-4-26b-a4b-it",
    )
});
