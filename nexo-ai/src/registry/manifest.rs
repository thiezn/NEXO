use crate::download::{Component, ModelFile, ModelManifest};
use crate::shared::types::ModelCategory;
use std::sync::LazyLock;

/// Component types for AI model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiComponent {
    Model,
    ModelShard,
    Tokenizer,
    Config,
    Vae,
    TextEncoder,
    ClipEncoder,
    T5Encoder,
    VisionProjector,
}

impl Component for AiComponent {
    fn name(&self) -> &str {
        match self {
            Self::Model => "model",
            Self::ModelShard => "model_shard",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::ClipEncoder => "clip_encoder",
            Self::T5Encoder => "t5_encoder",
            Self::VisionProjector => "vision_projector",
        }
    }

    fn is_model_specific(&self) -> bool {
        match self {
            Self::Model
            | Self::ModelShard
            | Self::Tokenizer
            | Self::Config
            | Self::VisionProjector => true,
            Self::Vae | Self::TextEncoder | Self::ClipEncoder | Self::T5Encoder => false,
        }
    }
}

/// An AI model manifest with associated categories.
pub struct AiModelManifest {
    pub manifest: ModelManifest<AiComponent>,
    pub categories: Vec<ModelCategory>,
    /// HuggingFace repo identifier for remote-served models (e.g. MLX).
    pub hf_repo: Option<String>,
}

// ── Whisper ────────────────────────────────────────────────────────────────

fn whisper_large_v3_manifest() -> AiModelManifest {
    let repo = "openai/whisper-large-v3".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "whisper-large-v3".to_string(),
            family: "whisper".to_string(),
            description: "Whisper Large v3 — high accuracy transcription (~2.9 GB)".to_string(),
            size_gb: 2.9,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "model.safetensors".to_string(),
                    size_bytes: 3_087_130_976,
                    gated: false,
                    sha256: Some(
                        "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 2_480_617,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 1_272,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Listen],
        hf_repo: None,
    }
}

fn whisper_large_v3_turbo_manifest() -> AiModelManifest {
    let repo = "openai/whisper-large-v3-turbo".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "whisper-large-v3-turbo".to_string(),
            family: "whisper".to_string(),
            description: "Whisper Large v3 Turbo — fast transcription, 4 decoder layers (~1.5 GB)"
                .to_string(),
            size_gb: 1.5,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "model.safetensors".to_string(),
                    size_bytes: 1_617_824_864,
                    gated: false,
                    sha256: Some(
                        "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 2_710_337,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 1_256,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Listen],
        hf_repo: None,
    }
}

fn distil_large_v3_manifest() -> AiModelManifest {
    let repo = "distil-whisper/distil-large-v3".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "distil-large-v3".to_string(),
            family: "whisper".to_string(),
            description: "Distil-Whisper Large v3 — distilled, 2 decoder layers (~1.4 GB)"
                .to_string(),
            size_gb: 1.4,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "model.safetensors".to_string(),
                    size_bytes: 1_512_874_472,
                    gated: false,
                    sha256: Some(
                        "065e3775409aa2fb0e6893a91f7210d3f4c930536ab79acaa56dafbf7be3a475",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 2_480_617,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 1_372,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Listen],
        hf_repo: None,
    }
}

// ── Flux.2 ─────────────────────────────────────────────────────────────────

fn flux_2_klein_4b_manifest() -> AiModelManifest {
    let repo = "black-forest-labs/FLUX.2-klein-4b".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "flux-2-klein-4b".to_string(),
            family: "flux".to_string(),
            description: "Flux.2 Klein 4B — fast 4-step image generation (~12 GB)".to_string(),
            size_gb: 12.0,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 7_751_109_744,
                    gated: false,
                    sha256: Some(
                        "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00002.safetensors".to_string(),
                    size_bytes: 4_967_215_360,
                    gated: false,
                    sha256: Some(
                        "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00002.safetensors".to_string(),
                    size_bytes: 3_077_766_632,
                    gated: false,
                    sha256: Some(
                        "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07",
                    ),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 168_120_878,
                    gated: false,
                    sha256: Some(
                        "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

fn flux_2_klein_9b_manifest() -> AiModelManifest {
    let repo = "black-forest-labs/FLUX.2-klein-9b".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "flux-2-klein-9b".to_string(),
            family: "flux".to_string(),
            description: "Flux.2 Klein 9B — high quality 4-step image generation (~35 GB)"
                .to_string(),
            size_gb: 34.7,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                        .to_string(),
                    size_bytes: 9_801_069_272,
                    gated: true,
                    sha256: Some(
                        "cb942a7072865a1d06e47a3361f9ba8746e68ad207c8499083bcb735869f5102",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                        .to_string(),
                    size_bytes: 8_356_121_608,
                    gated: true,
                    sha256: Some(
                        "ca568a31d19c03ddbcfd8b2d4ec7dbd16dcefbaa50b7aef1b8ceefd6e6eb0970",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
                    size_bytes: 4_902_257_696,
                    gated: true,
                    sha256: Some(
                        "c0dc64934ae0f730ddc80d99af44968d01a89e8454df07d762096ea1356446bc",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
                    size_bytes: 4_915_960_368,
                    gated: true,
                    sha256: Some(
                        "d58533b468c31caa0222540a8aefddea1d74dc6e1fee928da8556d3d85729d6e",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
                    size_bytes: 4_983_068_496,
                    gated: true,
                    sha256: Some(
                        "74927fec432e050365bf757bb30348f560a44394efac89f492680b7c910b64fd",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
                    size_bytes: 1_580_230_264,
                    gated: true,
                    sha256: Some(
                        "cb73f466fade5716702bda38d4e3b321c9358c39889e46fb9d613fb038bfcb2f",
                    ),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 168_120_878,
                    gated: true,
                    sha256: Some(
                        "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

fn flux_2_dev_manifest() -> AiModelManifest {
    let repo = "black-forest-labs/FLUX.2-dev".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "flux-2-dev".to_string(),
            family: "flux".to_string(),
            description: "Flux.2 Dev — full precision image generation (~165 GB)".to_string(),
            size_gb: 165.0,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_935_797_200,
                    gated: true,
                    sha256: Some(
                        "9d9b85f75f72fb17c7d29dacf7c430e924da93122d578a559a36a7635e153714",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_890_181_048,
                    gated: true,
                    sha256: Some(
                        "86adf6f41474b00bd57afbb29a09f008be7d6af8ae914956585ba5bc6bf97c28",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_480,
                    gated: true,
                    sha256: Some(
                        "a14e26e8f305dd26d7881f333e6e6ce5b562cbb55282538f46d38e1ff2715179",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some(
                        "5c4f38976fd8d7e5fb2d4cd20562d74eebba3264566987e3ef938d807c75be90",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some(
                        "4d7a74d916fc22117cde8bad76aa4b561e6dc92368cddc23bdae06dbc586ad95",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some(
                        "08f3ad03610651f9d630177ac3a4770d532fa72d788d3b36c39a0301b1595447",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 5_361_898_792,
                    gated: true,
                    sha256: Some(
                        "789b9bacb607e9b97597f77c86056fa6cbb747c2a6016588e6e196814b5f9733",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00010.safetensors".to_string(),
                    size_bytes: 4_883_550_696,
                    gated: true,
                    sha256: Some(
                        "91831c2ce219df0ce63bc33c6249e5cb01db8d93816bcebf975f1c406286520e",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_336,
                    gated: true,
                    sha256: Some(
                        "8ffe80706a66b2f5ef1fb058806ccf09f124ec4ad38af7a377e44ab1ee2fd664",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00003-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_224,
                    gated: true,
                    sha256: Some(
                        "99ec66e891f9563f568734eadfc5b7701e04620e8e163d4d5755277a3b50cf2f",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00004-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_376,
                    gated: true,
                    sha256: Some(
                        "e1df1527b12b1eb5cbd9a50914f9e6eb24e885ec830a3c16b5eed6ad0b53a396",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00005-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_368,
                    gated: true,
                    sha256: Some(
                        "3556ac03f47c24eb8ad27c237e25baad639c651d9596fd72cb1523137bf56163",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00006-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_248,
                    gated: true,
                    sha256: Some(
                        "2c41e6f80f2b5ca384ce703eac048a13daf2aff689c3acca66a8943f45338aae",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00007-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_376,
                    gated: true,
                    sha256: Some(
                        "62a725f154f6ba942a36b5cc450db2b2df32f434e3224558c789bc04fa05fd36",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00008-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_368,
                    gated: true,
                    sha256: Some(
                        "3a1a6ac77e6434418bb7273b68a7b3534fed5217c990061c92a8f990dd6ab20e",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00009-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_248,
                    gated: true,
                    sha256: Some(
                        "e1fffc9bb2b77d4d2382c1bd9053e9d017741d67ca00cc6f77034a294f2f5cfd",
                    ),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00010-of-00010.safetensors".to_string(),
                    size_bytes: 4_571_866_320,
                    gated: true,
                    sha256: Some(
                        "116ef7ae6fa0fd46b478324e4aa6a49f448afed900ca9f71d4fbd3d02289bbd4",
                    ),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 336_213_556,
                    gated: true,
                    sha256: Some(
                        "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

// ── Z-Image ────────────────────────────────────────────────────────────────

fn z_image_turbo_manifest() -> AiModelManifest {
    let orig_repo = "Tongyi-MAI/Z-Image-Turbo".to_string();
    let gguf_repo = "jayn7/Z-Image-Turbo-GGUF".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "z-image-turbo".to_string(),
            family: "z_image".to_string(),
            description:
                "Z-Image Turbo — 6B text-to-image, Q4_K_M quantized for Apple Silicon (~12 GB)"
                    .to_string(),
            size_gb: 11.9,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: gguf_repo,
                    hf_filename: "z_image_turbo-Q4_K_M.gguf".to_string(),
                    size_bytes: 3_692_871_680,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Vae,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 167_666_902,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::TextEncoder,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00003.safetensors".to_string(),
                    size_bytes: 3_957_900_840,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::TextEncoder,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00003.safetensors".to_string(),
                    size_bytes: 3_987_450_520,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::TextEncoder,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "text_encoder/model-00003-of-00003.safetensors".to_string(),
                    size_bytes: 99_630_640,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: orig_repo,
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: Some(
                        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                    ),
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

// ── Qwen-Image ─────────────────────────────────────────────────────────────

/// Shared VAE + text encoder + tokenizer files for all Qwen-Image variants.
fn qwen_image_shared_files() -> Vec<ModelFile<AiComponent>> {
    let qwen_repo = "Qwen/Qwen-Image-2512".to_string();
    vec![
        ModelFile {
            component: AiComponent::Vae,
            hf_repo: qwen_repo.clone(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            size_bytes: 253_806_966,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::TextEncoder,
            hf_repo: qwen_repo.clone(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            size_bytes: 4_968_243_304,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::TextEncoder,
            hf_repo: qwen_repo.clone(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            size_bytes: 4_991_495_816,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::TextEncoder,
            hf_repo: qwen_repo.clone(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            size_bytes: 4_932_751_040,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::TextEncoder,
            hf_repo: qwen_repo,
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            size_bytes: 1_691_924_384,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::Tokenizer,
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            size_bytes: 7_031_645,
            gated: false,
            sha256: None,
        },
    ]
}

fn qwen_image_bf16_manifest() -> AiModelManifest {
    let repo = "Qwen/Qwen-Image-2512".to_string();
    let mut files = vec![
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00001-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_989_364_312,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00002-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_984_214_160,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00003-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_946_470_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00004-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_984_213_736,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00005-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_946_471_896,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00006-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_946_451_560,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00007-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_908_690_520,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.clone(),
            hf_filename: "transformer/diffusion_pytorch_model-00008-of-00009.safetensors"
                .to_string(),
            size_bytes: 4_984_232_856,
            gated: false,
            sha256: None,
        },
        ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo,
            hf_filename: "transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
                .to_string(),
            size_bytes: 1_170_918_840,
            gated: false,
            sha256: None,
        },
    ];
    files.extend(qwen_image_shared_files());
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen-image-bf16".to_string(),
            family: "qwen_image".to_string(),
            description: "Qwen-Image BF16 — full precision text-to-image (~54 GB)".to_string(),
            size_gb: 53.7,
            files,
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

fn qwen_image_q8_manifest() -> AiModelManifest {
    let mut files = vec![ModelFile {
        component: AiComponent::Model,
        hf_repo: "city96/Qwen-Image-gguf".to_string(),
        hf_filename: "qwen-image-Q8_0.gguf".to_string(),
        size_bytes: 21_761_817_120,
        gated: false,
        sha256: None,
    }];
    files.extend(qwen_image_shared_files());
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen-image-q8".to_string(),
            family: "qwen_image".to_string(),
            description: "Qwen-Image Q8_0 — high quality quantized text-to-image (~38 GB)"
                .to_string(),
            size_gb: 36.0,
            files,
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

fn qwen_image_q6_manifest() -> AiModelManifest {
    let mut files = vec![ModelFile {
        component: AiComponent::Model,
        hf_repo: "city96/Qwen-Image-gguf".to_string(),
        hf_filename: "qwen-image-Q6_K.gguf".to_string(),
        size_bytes: 16_824_990_240,
        gated: false,
        sha256: None,
    }];
    files.extend(qwen_image_shared_files());
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen-image-q6".to_string(),
            family: "qwen_image".to_string(),
            description: "Qwen-Image Q6_K — balanced quantized text-to-image (~33 GB)".to_string(),
            size_gb: 31.3,
            files,
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

fn qwen_image_q4_manifest() -> AiModelManifest {
    let mut files = vec![ModelFile {
        component: AiComponent::Model,
        hf_repo: "city96/Qwen-Image-gguf".to_string(),
        hf_filename: "qwen-image-Q4_K_S.gguf".to_string(),
        size_bytes: 12_140_608_032,
        gated: false,
        sha256: None,
    }];
    files.extend(qwen_image_shared_files());
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen-image-q4".to_string(),
            family: "qwen_image".to_string(),
            description: "Qwen-Image Q4_K_S — fast quantized text-to-image (~29 GB)".to_string(),
            size_gb: 27.0,
            files,
        },
        categories: vec![ModelCategory::Imagine],
        hf_repo: None,
    }
}

// ── Gemma 4 ────────────────────────────────────────────────────────────────

fn gemma4_manifest(
    name: &str,
    repo: &str,
    description: &str,
    size_gb: f32,
    model_files: Vec<ModelFile<AiComponent>>,
) -> AiModelManifest {
    let mut files = model_files;
    files.push(ModelFile {
        component: AiComponent::Tokenizer,
        hf_repo: repo.to_string(),
        hf_filename: "tokenizer.json".to_string(),
        size_bytes: 32_169_626,
        gated: false,
        sha256: None,
    });
    files.push(ModelFile {
        component: AiComponent::Config,
        hf_repo: repo.to_string(),
        hf_filename: "config.json".to_string(),
        size_bytes: 4_954,
        gated: false,
        sha256: None,
    });
    AiModelManifest {
        manifest: ModelManifest {
            name: name.to_string(),
            family: "gemma4".to_string(),
            description: description.to_string(),
            size_gb,
            files,
        },
        categories: vec![
            ModelCategory::Chat,
            ModelCategory::Tool,
            ModelCategory::Image,
        ],
        hf_repo: None,
    }
}

fn gemma_4_e2b_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-E2B";
    gemma4_manifest(
        "gemma-4-e2b",
        repo,
        "Gemma 4 E2B — compact multimodal base model (~9.5 GB)",
        9.5,
        vec![ModelFile {
            component: AiComponent::Model,
            hf_repo: repo.to_string(),
            hf_filename: "model.safetensors".to_string(),
            size_bytes: 10_246_621_918,
            gated: false,
            sha256: None,
        }],
    )
}

fn gemma_4_e2b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-E2B-it";
    gemma4_manifest(
        "gemma-4-e2b-it",
        repo,
        "Gemma 4 E2B IT — compact multimodal chat, tool & vision (~9.5 GB)",
        9.5,
        vec![ModelFile {
            component: AiComponent::Model,
            hf_repo: repo.to_string(),
            hf_filename: "model.safetensors".to_string(),
            size_bytes: 10_246_621_918,
            gated: false,
            sha256: None,
        }],
    )
}

fn gemma_4_e4b_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-E4B";
    gemma4_manifest(
        "gemma-4-e4b",
        repo,
        "Gemma 4 E4B — mid-size multimodal base model (~14.9 GB)",
        14.9,
        vec![ModelFile {
            component: AiComponent::Model,
            hf_repo: repo.to_string(),
            hf_filename: "model.safetensors".to_string(),
            size_bytes: 15_992_595_884,
            gated: false,
            sha256: None,
        }],
    )
}

fn gemma_4_e4b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-E4B-it";
    gemma4_manifest(
        "gemma-4-e4b-it",
        repo,
        "Gemma 4 E4B IT — mid-size multimodal chat, tool & vision (~14.9 GB)",
        14.9,
        vec![ModelFile {
            component: AiComponent::Model,
            hf_repo: repo.to_string(),
            hf_filename: "model.safetensors".to_string(),
            size_bytes: 15_992_595_884,
            gated: false,
            sha256: None,
        }],
    )
}

fn gemma_4_26b_a4b_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-26b-a4b";
    gemma4_manifest(
        "gemma-4-26b-a4b",
        repo,
        "Gemma 4 26B-A4B — MoE multimodal base model (~48 GB)",
        48.1,
        vec![
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00001-of-00002.safetensors".to_string(),
                size_bytes: 49_907_246_508,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00002-of-00002.safetensors".to_string(),
                size_bytes: 1_704_763_408,
                gated: false,
                sha256: None,
            },
        ],
    )
}

fn gemma_4_26b_a4b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-26b-a4b-it";
    gemma4_manifest(
        "gemma-4-26b-a4b-it",
        repo,
        "Gemma 4 26B-A4B IT — MoE multimodal chat, tool & vision (~48 GB)",
        48.1,
        vec![
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00001-of-00002.safetensors".to_string(),
                size_bytes: 49_907_246_508,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00002-of-00002.safetensors".to_string(),
                size_bytes: 1_704_763_408,
                gated: false,
                sha256: None,
            },
        ],
    )
}

fn gemma_4_31b_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-31b";
    gemma4_manifest(
        "gemma-4-31b",
        repo,
        "Gemma 4 31B — large multimodal base model (~58 GB)",
        58.3,
        vec![
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00001-of-00002.safetensors".to_string(),
                size_bytes: 49_784_788_364,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00002-of-00002.safetensors".to_string(),
                size_bytes: 12_761_549_884,
                gated: false,
                sha256: None,
            },
        ],
    )
}

fn gemma_4_31b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-4-31b-it";
    gemma4_manifest(
        "gemma-4-31b-it",
        repo,
        "Gemma 4 31B IT — large multimodal chat, tool & vision (~58 GB)",
        58.3,
        vec![
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00001-of-00002.safetensors".to_string(),
                size_bytes: 49_784_788_364,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: AiComponent::ModelShard,
                hf_repo: repo.to_string(),
                hf_filename: "model-00002-of-00002.safetensors".to_string(),
                size_bytes: 12_761_549_884,
                gated: false,
                sha256: None,
            },
        ],
    )
}

// ── Gemma 4 GGUF ──────────────────────────────────────────────────────────

fn gemma_4_e2b_it_q5_manifest() -> AiModelManifest {
    let gguf_repo = "unsloth/gemma-4-e2b-it-GGUF".to_string();
    let orig_repo = "google/gemma-4-E2B-it".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "gemma-4-e2b-it-q5".to_string(),
            family: "gemma4".to_string(),
            description:
                "Gemma 4 E2B IT Q5_K_M — quantized multimodal chat, tool & vision (~4.1 GB)"
                    .to_string(),
            size_gb: 4.1,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: gguf_repo.clone(),
                    hf_filename: "gemma-4-E2B-it-Q5_K_M.gguf".to_string(),
                    size_bytes: 3_356_030_336,
                    gated: false,
                    sha256: None,
                },
                // Multimodal projector (vision/audio towers) in GGUF F16 format
                ModelFile {
                    component: AiComponent::VisionProjector,
                    hf_repo: gguf_repo,
                    hf_filename: "mmproj-F16.gguf".to_string(),
                    size_bytes: 985_654_208,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 32_169_626,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: orig_repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 4_954,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![
            ModelCategory::Chat,
            ModelCategory::Tool,
            ModelCategory::Image,
            ModelCategory::Listen,
        ],
        hf_repo: None,
    }
}

fn gemma_4_26b_a4b_it_q4_manifest() -> AiModelManifest {
    let gguf_repo = "unsloth/gemma-4-26b-a4b-it-GGUF".to_string();
    let orig_repo = "google/gemma-4-26b-a4b-it".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "gemma-4-26b-a4b-it-q4".to_string(),
            family: "gemma4".to_string(),
            description:
                "Gemma 4 26B-A4B IT UD-Q4_K_M — quantized MoE multimodal chat, tool & vision (~17.0 GB)"
                    .to_string(),
            size_gb: 17.0,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: gguf_repo.clone(),
                    hf_filename: "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
                    size_bytes: 16_868_236_288,
                    gated: false,
                    sha256: None,
                },
                // Multimodal projector (vision tower) in GGUF F16 format
                ModelFile {
                    component: AiComponent::VisionProjector,
                    hf_repo: gguf_repo,
                    hf_filename: "mmproj-F16.gguf".to_string(),
                    size_bytes: 1_193_058_912,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: orig_repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 32_169_626,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: orig_repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 4_954,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![
            ModelCategory::Chat,
            ModelCategory::Tool,
            ModelCategory::Image,
        ],
        hf_repo: None,
    }
}

// ── MLX (mlx_vlm server) ─────────────────────────────────────────────────

#[cfg(feature = "mlx")]
fn mlx_manifest(
    name: &str,
    repo: &str,
    description: &str,
    size_gb: f32,
    shard_sizes: &[u64],
    config_sizes: MlxConfigSizes,
) -> AiModelManifest {
    let num_shards = shard_sizes.len();
    let mut files: Vec<ModelFile<AiComponent>> = shard_sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| ModelFile {
            component: AiComponent::ModelShard,
            hf_repo: repo.to_string(),
            hf_filename: format!("model-{:05}-of-{:05}.safetensors", i + 1, num_shards),
            size_bytes: size,
            gated: false,
            sha256: None,
        })
        .collect();

    for (filename, size) in [
        ("config.json", config_sizes.config),
        ("model.safetensors.index.json", config_sizes.index),
        ("processor_config.json", config_sizes.processor),
        ("tokenizer_config.json", config_sizes.tokenizer_config),
        ("generation_config.json", config_sizes.generation),
    ] {
        files.push(ModelFile {
            component: AiComponent::Config,
            hf_repo: repo.to_string(),
            hf_filename: filename.to_string(),
            size_bytes: size,
            gated: false,
            sha256: None,
        });
    }

    files.push(ModelFile {
        component: AiComponent::Tokenizer,
        hf_repo: repo.to_string(),
        hf_filename: "tokenizer.json".to_string(),
        size_bytes: config_sizes.tokenizer,
        gated: false,
        sha256: None,
    });

    AiModelManifest {
        manifest: ModelManifest {
            name: name.to_string(),
            family: "mlx".to_string(),
            description: description.to_string(),
            size_gb,
            files,
        },
        categories: vec![
            ModelCategory::Chat,
            ModelCategory::Tool,
            ModelCategory::Image,
        ],
        hf_repo: Some(repo.to_string()),
    }
}

#[cfg(feature = "mlx")]
struct MlxConfigSizes {
    config: u64,
    index: u64,
    processor: u64,
    tokenizer_config: u64,
    generation: u64,
    tokenizer: u64,
}

#[cfg(feature = "mlx")]
fn mlx_gemma_4_e2b_it_8bit_manifest() -> AiModelManifest {
    mlx_manifest(
        "mlx-gemma-4-e2b-it-8bit",
        "mlx-community/gemma-4-E2B-it-8bit",
        "Gemma 4 E2B IT 8-bit via MLX — multimodal chat, tool & vision (~5.5 GB)",
        5.5,
        &[5_367_135_201, 532_432_577],
        MlxConfigSizes {
            config: 5_996,
            index: 270_064,
            processor: 902,
            tokenizer_config: 2_685,
            generation: 208,
            tokenizer: 32_169_626,
        },
    )
}

#[cfg(feature = "mlx")]
fn mlx_gemma_4_26b_a4b_4bit_manifest() -> AiModelManifest {
    mlx_manifest(
        "mlx-gemma-4-26b-a4b-4bit",
        "mlx-community/gemma-4-26b-a4b-4bit",
        "Gemma 4 26B-A4B 4-bit via MLX — MoE multimodal chat, tool & vision (~15.6 GB)",
        15.6,
        &[5_275_612_587, 5_296_718_232, 5_036_507_755],
        MlxConfigSizes {
            config: 33_340,
            index: 176_940,
            processor: 627,
            tokenizer_config: 1_498,
            generation: 181,
            tokenizer: 32_170_070,
        },
    )
}

#[cfg(feature = "mlx")]
fn mlx_gemma_4_31b_it_4bit_manifest() -> AiModelManifest {
    mlx_manifest(
        "mlx-gemma-4-31b-it-4bit",
        "mlx-community/gemma-4-31b-it-4bit",
        "Gemma 4 31B IT 4-bit via MLX — large multimodal chat, tool & vision (~18.4 GB)",
        18.4,
        &[5_366_617_542, 5_361_642_569, 5_367_276_098, 2_316_480_623],
        MlxConfigSizes {
            config: 5_647,
            index: 205_370,
            processor: 627,
            tokenizer_config: 2_685,
            generation: 208,
            tokenizer: 32_169_626,
        },
    )
}

// ── Registry ───────────────────────────────────────────────────────────────

fn build_all_manifests() -> Vec<AiModelManifest> {
    #[allow(unused_mut)]
    let mut all = vec![
        // Listen
        whisper_large_v3_manifest(),
        whisper_large_v3_turbo_manifest(),
        distil_large_v3_manifest(),
        // Imagine
        flux_2_klein_4b_manifest(),
        flux_2_klein_9b_manifest(),
        flux_2_dev_manifest(),
        z_image_turbo_manifest(),
        qwen_image_bf16_manifest(),
        qwen_image_q8_manifest(),
        qwen_image_q6_manifest(),
        qwen_image_q4_manifest(),
        // Chat + Tool + Image (safetensors)
        gemma_4_e2b_manifest(),
        gemma_4_e2b_it_manifest(),
        gemma_4_e4b_manifest(),
        gemma_4_e4b_it_manifest(),
        gemma_4_26b_a4b_manifest(),
        gemma_4_26b_a4b_it_manifest(),
        gemma_4_31b_manifest(),
        gemma_4_31b_it_manifest(),
        // Chat + Tool (GGUF quantized)
        gemma_4_e2b_it_q5_manifest(),
        gemma_4_26b_a4b_it_q4_manifest(),
    ];

    // MLX models (served via mlx_vlm server)
    #[cfg(feature = "mlx")]
    {
        all.push(mlx_gemma_4_e2b_it_8bit_manifest());
        all.push(mlx_gemma_4_26b_a4b_4bit_manifest());
        all.push(mlx_gemma_4_31b_it_4bit_manifest());
    }

    all
}

static ALL_MANIFESTS: LazyLock<Vec<AiModelManifest>> = LazyLock::new(build_all_manifests);

/// Return all known AI model manifests.
pub fn known_manifests() -> &'static [AiModelManifest] {
    &ALL_MANIFESTS
}

/// Look up a manifest by name (case-sensitive).
pub fn find_manifest(name: &str) -> Option<&'static AiModelManifest> {
    ALL_MANIFESTS.iter().find(|m| m.manifest.name == name)
}

/// Return all manifests that belong to a given category.
pub fn manifests_for_category(category: ModelCategory) -> Vec<&'static AiModelManifest> {
    ALL_MANIFESTS
        .iter()
        .filter(|m| m.categories.contains(&category))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ai_component_names() {
        assert_eq!(AiComponent::Model.name(), "model");
        assert_eq!(AiComponent::ModelShard.name(), "model_shard");
        assert_eq!(AiComponent::Tokenizer.name(), "tokenizer");
        assert_eq!(AiComponent::Config.name(), "config");
        assert_eq!(AiComponent::Vae.name(), "vae");
        assert_eq!(AiComponent::TextEncoder.name(), "text_encoder");
        assert_eq!(AiComponent::ClipEncoder.name(), "clip_encoder");
        assert_eq!(AiComponent::T5Encoder.name(), "t5_encoder");
        assert_eq!(AiComponent::VisionProjector.name(), "vision_projector");
    }

    #[test]
    fn ai_component_model_specificity() {
        assert!(AiComponent::Model.is_model_specific());
        assert!(AiComponent::ModelShard.is_model_specific());
        assert!(AiComponent::Tokenizer.is_model_specific());
        assert!(AiComponent::Config.is_model_specific());

        assert!(!AiComponent::Vae.is_model_specific());
        assert!(AiComponent::VisionProjector.is_model_specific());

        assert!(!AiComponent::TextEncoder.is_model_specific());
        assert!(!AiComponent::ClipEncoder.is_model_specific());
        assert!(!AiComponent::T5Encoder.is_model_specific());
    }

    #[test]
    fn known_manifests_count() {
        let expected = if cfg!(feature = "mlx") { 24 } else { 21 };
        assert_eq!(known_manifests().len(), expected);
    }

    #[test]
    fn known_manifests_contains_whisper() {
        let manifests = known_manifests();
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "whisper-large-v3")
        );
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "whisper-large-v3-turbo")
        );
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "distil-large-v3")
        );
    }

    #[test]
    fn whisper_manifests_are_listen_category() {
        for name in [
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "distil-large-v3",
        ] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "whisper");
            assert!(m.categories.contains(&ModelCategory::Listen));
        }
    }

    #[test]
    fn manifests_for_listen_contains_whisper() {
        let listen = manifests_for_category(ModelCategory::Listen);
        assert_eq!(listen.len(), 4); // 3 whisper + 1 gemma-4-e2b-it-q5
        assert!(
            listen
                .iter()
                .any(|m| m.manifest.name == "whisper-large-v3-turbo")
        );
        assert!(
            listen
                .iter()
                .any(|m| m.manifest.name == "gemma-4-e2b-it-q5")
        );
    }

    #[test]
    fn find_manifest_returns_none_for_unknown() {
        assert!(find_manifest("nonexistent-model").is_none());
    }

    #[test]
    fn known_manifests_contains_flux() {
        let manifests = known_manifests();
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "flux-2-klein-4b")
        );
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "flux-2-klein-9b")
        );
        assert!(manifests.iter().any(|m| m.manifest.name == "flux-2-dev"));
    }

    #[test]
    fn flux_manifests_are_imagine_category() {
        for name in ["flux-2-klein-4b", "flux-2-klein-9b", "flux-2-dev"] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "flux");
            assert!(m.categories.contains(&ModelCategory::Imagine));
        }
    }

    #[test]
    fn flux_klein_4b_is_ungated() {
        let m = find_manifest("flux-2-klein-4b").unwrap();
        assert!(m.manifest.files.iter().all(|f| !f.gated));
    }

    #[test]
    fn flux_klein_9b_is_gated() {
        let m = find_manifest("flux-2-klein-9b").unwrap();
        assert!(m.manifest.files.iter().any(|f| f.gated));
    }

    #[test]
    fn flux_dev_is_gated() {
        let m = find_manifest("flux-2-dev").unwrap();
        assert!(m.manifest.files.iter().any(|f| f.gated));
    }

    #[test]
    fn z_image_turbo_manifest_has_all_components() {
        let m = find_manifest("z-image-turbo").unwrap();
        assert_eq!(m.manifest.family, "z_image");
        assert!(m.categories.contains(&ModelCategory::Imagine));
        assert!(
            m.manifest
                .files
                .iter()
                .any(|f| f.component == AiComponent::Model)
        );
        assert!(
            m.manifest
                .files
                .iter()
                .any(|f| f.component == AiComponent::Vae)
        );
        assert!(
            m.manifest
                .files
                .iter()
                .any(|f| f.component == AiComponent::TextEncoder)
        );
        assert!(
            m.manifest
                .files
                .iter()
                .any(|f| f.component == AiComponent::Tokenizer)
        );
    }

    #[test]
    fn manifests_for_imagine_contains_all() {
        let imagine = manifests_for_category(ModelCategory::Imagine);
        assert_eq!(imagine.len(), 8); // 3 flux + 1 z_image + 4 qwen_image
        assert!(imagine.iter().any(|m| m.manifest.name == "flux-2-klein-4b"));
        assert!(imagine.iter().any(|m| m.manifest.name == "z-image-turbo"));
        assert!(imagine.iter().any(|m| m.manifest.name == "qwen-image-q4"));
    }

    #[test]
    fn known_manifests_contains_qwen_image() {
        let manifests = known_manifests();
        assert!(
            manifests
                .iter()
                .any(|m| m.manifest.name == "qwen-image-bf16")
        );
        assert!(manifests.iter().any(|m| m.manifest.name == "qwen-image-q8"));
        assert!(manifests.iter().any(|m| m.manifest.name == "qwen-image-q6"));
        assert!(manifests.iter().any(|m| m.manifest.name == "qwen-image-q4"));
    }

    #[test]
    fn qwen_image_manifests_are_imagine_category() {
        for name in [
            "qwen-image-bf16",
            "qwen-image-q8",
            "qwen-image-q6",
            "qwen-image-q4",
        ] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "qwen_image");
            assert!(m.categories.contains(&ModelCategory::Imagine));
        }
    }

    #[test]
    fn qwen_image_models_are_ungated() {
        for name in [
            "qwen-image-bf16",
            "qwen-image-q8",
            "qwen-image-q6",
            "qwen-image-q4",
        ] {
            let m = find_manifest(name).unwrap();
            assert!(m.manifest.files.iter().all(|f| !f.gated));
        }
    }

    #[test]
    fn qwen_image_has_shared_components() {
        for name in [
            "qwen-image-bf16",
            "qwen-image-q8",
            "qwen-image-q6",
            "qwen-image-q4",
        ] {
            let m = find_manifest(name).unwrap();
            assert!(
                m.manifest
                    .files
                    .iter()
                    .any(|f| f.component == AiComponent::Vae)
            );
            assert!(
                m.manifest
                    .files
                    .iter()
                    .any(|f| f.component == AiComponent::TextEncoder)
            );
            assert!(
                m.manifest
                    .files
                    .iter()
                    .any(|f| f.component == AiComponent::Tokenizer)
            );
        }
    }

    #[test]
    fn known_manifests_contains_gemma4() {
        let manifests = known_manifests();
        for name in [
            "gemma-4-e2b",
            "gemma-4-e2b-it",
            "gemma-4-e4b",
            "gemma-4-e4b-it",
            "gemma-4-26b-a4b",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b",
            "gemma-4-31b-it",
        ] {
            assert!(
                manifests.iter().any(|m| m.manifest.name == name),
                "missing: {name}"
            );
        }
    }

    #[test]
    fn gemma4_manifests_are_multipurpose() {
        for name in [
            "gemma-4-e2b",
            "gemma-4-e2b-it",
            "gemma-4-e4b",
            "gemma-4-e4b-it",
            "gemma-4-26b-a4b",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b",
            "gemma-4-31b-it",
        ] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "gemma4");
            assert!(m.categories.contains(&ModelCategory::Chat));
            assert!(m.categories.contains(&ModelCategory::Tool));
            assert!(m.categories.contains(&ModelCategory::Image));
        }
    }

    #[test]
    fn known_manifests_contains_gemma4_gguf() {
        let manifests = known_manifests();
        for name in ["gemma-4-e2b-it-q5", "gemma-4-26b-a4b-it-q4"] {
            assert!(
                manifests.iter().any(|m| m.manifest.name == name),
                "missing: {name}"
            );
        }
    }

    #[test]
    fn gemma4_gguf_manifests_have_multimodal_categories() {
        // E2B GGUF: Chat + Tool + Image + Listen
        let e2b = find_manifest("gemma-4-e2b-it-q5").unwrap();
        assert_eq!(e2b.manifest.family, "gemma4");
        assert!(e2b.categories.contains(&ModelCategory::Chat));
        assert!(e2b.categories.contains(&ModelCategory::Tool));
        assert!(e2b.categories.contains(&ModelCategory::Image));
        assert!(e2b.categories.contains(&ModelCategory::Listen));

        // 26B GGUF: Chat + Tool + Image (no audio)
        let big = find_manifest("gemma-4-26b-a4b-it-q4").unwrap();
        assert_eq!(big.manifest.family, "gemma4");
        assert!(big.categories.contains(&ModelCategory::Chat));
        assert!(big.categories.contains(&ModelCategory::Tool));
        assert!(big.categories.contains(&ModelCategory::Image));
        assert!(!big.categories.contains(&ModelCategory::Listen));
    }

    #[test]
    fn gemma4_gguf_has_mmproj_and_config() {
        for name in ["gemma-4-e2b-it-q5", "gemma-4-26b-a4b-it-q4"] {
            let m = find_manifest(name).unwrap();
            assert!(
                m.manifest
                    .files
                    .iter()
                    .any(|f| f.component == AiComponent::Config),
                "{name} should include config.json for multimodal"
            );
            assert!(
                m.manifest
                    .files
                    .iter()
                    .any(|f| f.component == AiComponent::VisionProjector),
                "{name} should include mmproj GGUF for vision/audio towers"
            );
        }
    }

    #[test]
    fn gemma4_models_are_ungated() {
        for name in [
            "gemma-4-e2b",
            "gemma-4-e2b-it",
            "gemma-4-e4b",
            "gemma-4-e4b-it",
            "gemma-4-26b-a4b",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b",
            "gemma-4-31b-it",
        ] {
            let m = find_manifest(name).unwrap();
            assert!(m.manifest.files.iter().all(|f| !f.gated));
        }
    }

    #[test]
    fn manifests_for_chat_contains_gemma4() {
        let chat = manifests_for_category(ModelCategory::Chat);
        let expected = if cfg!(feature = "mlx") { 13 } else { 10 };
        assert_eq!(chat.len(), expected);
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-4-e2b-it"));
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-4-31b-it"));
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-4-e2b-it-q5"));
        assert!(
            chat.iter()
                .any(|m| m.manifest.name == "gemma-4-26b-a4b-it-q4")
        );
    }
}
