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
                "chat_template.jinja",
                17_336,
                "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
            ),
            ModelFile::new(
                ModelFileKind::Config,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "config.json",
                5_145,
                "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
            ),
            ModelFile::new(
                ModelFileKind::GenerationConfig,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "generation_config.json",
                208,
                "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
            ),
            ModelFile::new(
                ModelFileKind::ProcessorConfig,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "processor_config.json",
                1_689,
                "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
            ),
            ModelFile::new(
                ModelFileKind::Tokenizer,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "tokenizer.json",
                32_169_626,
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            ),
            ModelFile::new(
                ModelFileKind::UqffResidual,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "residual.safetensors",
                8_554_380_772,
                "7b19ee1557f88191c2f317a62a2a325be5d66c67610c60b78e83453de4930872",
            ),
            ModelFile::new(
                ModelFileKind::UqffShard,
                GEMMA_4_E4B_IT_UQFF_REPO,
                "afq8-0.uqff",
                4_279_637_812,
                "d13c1cac7d313e354367478202fffe1ee0a9c1f9fc16e0037b501b79d7a01f49",
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
                "chat_template.jinja",
                17_466,
                "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
            ),
            ModelFile::new(
                ModelFileKind::Config,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "config.json",
                3_815,
                "ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3",
            ),
            ModelFile::new(
                ModelFileKind::GenerationConfig,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "generation_config.json",
                208,
                "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
            ),
            ModelFile::new(
                ModelFileKind::ProcessorConfig,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "processor_config.json",
                1_689,
                "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
            ),
            ModelFile::new(
                ModelFileKind::Tokenizer,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "tokenizer.json",
                32_169_626,
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            ),
            ModelFile::new(
                ModelFileKind::UqffResidual,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "residual.safetensors",
                2_645_114_980,
                "ab84f64d06fc1e9bd955e52348292054974a80b08f3047584db70fbd73d962ca",
            ),
            ModelFile::new(
                ModelFileKind::UqffShard,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "afq8-0.uqff",
                10_730_814_820,
                "a90c9c685987375c09b38994cc332a17bf638893a08f618006b034714d2436bc",
            ),
            ModelFile::new(
                ModelFileKind::UqffShard,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "afq8-1.uqff",
                10_675_090_944,
                "22db12c454f0b04fcb55bdfaabe72988e1191539dc984eedf7ea25c8110d1b9b",
            ),
            ModelFile::new(
                ModelFileKind::UqffShard,
                GEMMA_4_26B_A4B_IT_UQFF_REPO,
                "afq8-2.uqff",
                4_607_783_264,
                "4d43d02901e010afa78ee92745859ee0083ef719e8548c7a677e1d5697c2add7",
            ),
        ],
        "gemma-4-26b-a4b-it",
    )
});
