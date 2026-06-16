use super::super::{ModelFile, ModelFileKind};
use crate::ModelManifest;
use nexo_core::ModelId;
use std::sync::LazyLock;

const KOKORO_HF_REPO: &str = "hexgrad/Kokoro-82M";

/// Kokoro 82M model.
pub static KOKORO_82M: LazyLock<ModelManifest> = LazyLock::new(|| {
    ModelManifest::new(
        ModelId::Kokoro82m,
        0.4,
        vec![
            ModelFile::new(
                ModelFileKind::Weights,
                KOKORO_HF_REPO,
                "kokoro-v1_0.pth",
                0,
                "placeholder-sha256",
            ),
            ModelFile::new(
                ModelFileKind::Config,
                KOKORO_HF_REPO,
                "config.json",
                0,
                "placeholder-sha256",
            ),
            ModelFile::new_with_prefix(
                ModelFileKind::Modules,
                KOKORO_HF_REPO,
                "voices/",
                0,
                "placeholder-sha256",
            ),
        ],
    )
});
