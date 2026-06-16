use super::super::{ModelFile, ModelFileKind};
use crate::ModelManifest;
use nexo_core::ModelId;
use std::sync::LazyLock;

const EMBEDDING_GEMMA_HF_REPO: &str = "google/embeddinggemma-300m";

/// EmbeddingGemma 300M model.
pub static EMBEDDING_GEMMA_300M: LazyLock<ModelManifest> = LazyLock::new(|| {
    ModelManifest::new(
        ModelId::EmbeddingGemma300m,
        1.0,
        vec![
            ModelFile::new_with_suffix(
                ModelFileKind::Weights,
                EMBEDDING_GEMMA_HF_REPO,
                ".safetensors",
                300_000_000,
                "placeholder-sha256",
            ),
            ModelFile::new_with_suffix(
                ModelFileKind::Config,
                EMBEDDING_GEMMA_HF_REPO,
                ".json",
                300_000_000,
                "placeholder-sha256",
            ),
        ],
    )
});
