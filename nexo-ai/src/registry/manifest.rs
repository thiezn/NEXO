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
            Self::Model | Self::ModelShard | Self::Tokenizer | Self::Config
            | Self::VisionProjector => true,
            Self::Vae | Self::TextEncoder | Self::ClipEncoder | Self::T5Encoder => false,
        }
    }
}

/// An AI model manifest with associated categories.
pub struct AiModelManifest {
    pub manifest: ModelManifest<AiComponent>,
    pub categories: Vec<ModelCategory>,
}

// ── Registry ────────────────────────────────────────────────────────────────

fn parler_mini_manifest() -> AiModelManifest {
    let repo = "parler-tts/parler-tts-mini-v1.1".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "parler-mini".to_string(),
            family: "parler".to_string(),
            description: "Parler-TTS Mini v1.1 — fast TTS (~3.5 GB)".to_string(),
            size_gb: 3.5,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "model.safetensors".to_string(),
                    size_bytes: 3_751_321_772,
                    gated: false,
                    sha256: Some("f85ed0a4953b28f0bd9d3cec9f0e035df2936ba97646f315f54b42bf6ba6d0f9"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 10_272_460,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 7_311,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Talk],
    }
}

fn parler_large_manifest() -> AiModelManifest {
    let repo = "parler-tts/parler-tts-large-v1".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "parler-large".to_string(),
            family: "parler".to_string(),
            description: "Parler-TTS Large v1 — high quality TTS (~8.7 GB)".to_string(),
            size_gb: 8.7,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00001-of-00002.safetensors".to_string(),
                    size_bytes: 4_984_365_952,
                    gated: false,
                    sha256: Some("c30d2151a8a9c3343b6998eeec019b46db9d17b84ef234d98ea6723941c2851c"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00002-of-00002.safetensors".to_string(),
                    size_bytes: 4_347_810_672,
                    gated: false,
                    sha256: Some("413db5baa97486c7447a4060bb2ced9235d1785d51ac981e0880eff2cb044ca5"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 2_422_234,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 7_722,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Talk],
    }
}

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
                    sha256: Some("a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
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
                    sha256: Some("542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
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
                    sha256: Some("065e3775409aa2fb0e6893a91f7210d3f4c930536ab79acaa56dafbf7be3a475"),
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
    }
}

fn flux_2_klein_4b_manifest() -> AiModelManifest {
    let repo = "black-forest-labs/FLUX.2-klein-4b".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "flux-2-klein-4b".to_string(),
            family: "flux".to_string(),
            description: "Flux.2 Klein 4B — fast 4-step image generation (~22 GB)".to_string(),
            size_gb: 22.0,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 7_751_109_744,
                    gated: false,
                    sha256: Some("9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00002.safetensors".to_string(),
                    size_bytes: 4_967_215_360,
                    gated: false,
                    sha256: Some("8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00002.safetensors".to_string(),
                    size_bytes: 3_077_766_632,
                    gated: false,
                    sha256: Some("82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07"),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 168_120_878,
                    gated: false,
                    sha256: Some("ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
    }
}

fn flux_2_klein_9b_manifest() -> AiModelManifest {
    let repo = "black-forest-labs/FLUX.2-klein-9b".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "flux-2-klein-9b".to_string(),
            family: "flux".to_string(),
            description: "Flux.2 Klein 9B — high quality 4-step image generation (~49 GB)"
                .to_string(),
            size_gb: 49.0,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                        .to_string(),
                    size_bytes: 9_801_069_272,
                    gated: true,
                    sha256: Some("cb942a7072865a1d06e47a3361f9ba8746e68ad207c8499083bcb735869f5102"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                        .to_string(),
                    size_bytes: 8_356_121_608,
                    gated: true,
                    sha256: Some("ca568a31d19c03ddbcfd8b2d4ec7dbd16dcefbaa50b7aef1b8ceefd6e6eb0970"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
                    size_bytes: 4_902_257_696,
                    gated: true,
                    sha256: Some("c0dc64934ae0f730ddc80d99af44968d01a89e8454df07d762096ea1356446bc"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
                    size_bytes: 4_915_960_368,
                    gated: true,
                    sha256: Some("d58533b468c31caa0222540a8aefddea1d74dc6e1fee928da8556d3d85729d6e"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
                    size_bytes: 4_983_068_496,
                    gated: true,
                    sha256: Some("74927fec432e050365bf757bb30348f560a44394efac89f492680b7c910b64fd"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
                    size_bytes: 1_580_230_264,
                    gated: true,
                    sha256: Some("cb73f466fade5716702bda38d4e3b321c9358c39889e46fb9d613fb038bfcb2f"),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 168_120_878,
                    gated: true,
                    sha256: Some("ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
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
                    sha256: Some("9d9b85f75f72fb17c7d29dacf7c430e924da93122d578a559a36a7635e153714"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_890_181_048,
                    gated: true,
                    sha256: Some("86adf6f41474b00bd57afbb29a09f008be7d6af8ae914956585ba5bc6bf97c28"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_480,
                    gated: true,
                    sha256: Some("a14e26e8f305dd26d7881f333e6e6ce5b562cbb55282538f46d38e1ff2715179"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some("5c4f38976fd8d7e5fb2d4cd20562d74eebba3264566987e3ef938d807c75be90"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some("4d7a74d916fc22117cde8bad76aa4b561e6dc92368cddc23bdae06dbc586ad95"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 9_814_681_536,
                    gated: true,
                    sha256: Some("08f3ad03610651f9d630177ac3a4770d532fa72d788d3b36c39a0301b1595447"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
                        .to_string(),
                    size_bytes: 5_361_898_792,
                    gated: true,
                    sha256: Some("789b9bacb607e9b97597f77c86056fa6cbb747c2a6016588e6e196814b5f9733"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00001-of-00010.safetensors".to_string(),
                    size_bytes: 4_883_550_696,
                    gated: true,
                    sha256: Some("91831c2ce219df0ce63bc33c6249e5cb01db8d93816bcebf975f1c406286520e"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00002-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_336,
                    gated: true,
                    sha256: Some("8ffe80706a66b2f5ef1fb058806ccf09f124ec4ad38af7a377e44ab1ee2fd664"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00003-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_224,
                    gated: true,
                    sha256: Some("99ec66e891f9563f568734eadfc5b7701e04620e8e163d4d5755277a3b50cf2f"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00004-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_376,
                    gated: true,
                    sha256: Some("e1df1527b12b1eb5cbd9a50914f9e6eb24e885ec830a3c16b5eed6ad0b53a396"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00005-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_368,
                    gated: true,
                    sha256: Some("3556ac03f47c24eb8ad27c237e25baad639c651d9596fd72cb1523137bf56163"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00006-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_248,
                    gated: true,
                    sha256: Some("2c41e6f80f2b5ca384ce703eac048a13daf2aff689c3acca66a8943f45338aae"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00007-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_376,
                    gated: true,
                    sha256: Some("62a725f154f6ba942a36b5cc450db2b2df32f434e3224558c789bc04fa05fd36"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00008-of-00010.safetensors".to_string(),
                    size_bytes: 4_781_593_368,
                    gated: true,
                    sha256: Some("3a1a6ac77e6434418bb7273b68a7b3534fed5217c990061c92a8f990dd6ab20e"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00009-of-00010.safetensors".to_string(),
                    size_bytes: 4_886_472_248,
                    gated: true,
                    sha256: Some("e1fffc9bb2b77d4d2382c1bd9053e9d017741d67ca00cc6f77034a294f2f5cfd"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "text_encoder/model-00010-of-00010.safetensors".to_string(),
                    size_bytes: 4_571_866_320,
                    gated: true,
                    sha256: Some("116ef7ae6fa0fd46b478324e4aa6a49f448afed900ca9f71d4fbd3d02289bbd4"),
                },
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: repo.clone(),
                    hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                    size_bytes: 336_213_556,
                    gated: true,
                    sha256: Some("d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo,
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Imagine],
    }
}

fn gemma_3_4b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-3-4b-it".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "gemma-3-4b-it".to_string(),
            family: "gemma3".to_string(),
            description: "Gemma 3 4B IT — fast multimodal chat, tool & vision (~8 GB)".to_string(),
            size_gb: 8.0,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00001-of-00002.safetensors".to_string(),
                    size_bytes: 4_961_251_752,
                    gated: true,
                    sha256: Some("eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00002-of-00002.safetensors".to_string(),
                    size_bytes: 3_639_026_128,
                    gated: true,
                    sha256: Some("fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 33_384_568,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 855,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image],
    }
}

fn gemma_3_12b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-3-12b-it".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "gemma-3-12b-it".to_string(),
            family: "gemma3".to_string(),
            description: "Gemma 3 12B IT — multimodal chat, tool & vision (~22.7 GB)".to_string(),
            size_gb: 22.7,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00001-of-00005.safetensors".to_string(),
                    size_bytes: 4_979_902_192,
                    gated: true,
                    sha256: Some("4847447e92599833e8dbaa3067cd201c3bb5c052efa91f11ba891e43234f7832"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00002-of-00005.safetensors".to_string(),
                    size_bytes: 4_931_296_592,
                    gated: true,
                    sha256: Some("891bd54eed03cba9ee1e705533a02a8217fcc29f356e4a1f53e5fd0d178883ad"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00003-of-00005.safetensors".to_string(),
                    size_bytes: 4_931_296_656,
                    gated: true,
                    sha256: Some("7cee411d9d57324e50ce064a192cc5a858276d508611b12fc599e0c9767112e0"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00004-of-00005.safetensors".to_string(),
                    size_bytes: 4_931_296_656,
                    gated: true,
                    sha256: Some("8bc75a29a730c9e743cad013feda3b0991a913fafe787c58a1c6e20afad97723"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00005-of-00005.safetensors".to_string(),
                    size_bytes: 4_601_000_928,
                    gated: true,
                    sha256: Some("ed14bd4908c98fed9f61e8cd410167e0846de9abd78e0452ab092072e5d9252d"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 33_384_568,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 916,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image],
    }
}

fn gemma_3_27b_it_manifest() -> AiModelManifest {
    let repo = "google/gemma-3-27b-it".to_string();
    AiModelManifest {
        manifest: ModelManifest {
            name: "gemma-3-27b-it".to_string(),
            family: "gemma3".to_string(),
            description: "Gemma 3 27B IT — high quality multimodal chat, tool & vision (~51.1 GB)"
                .to_string(),
            size_gb: 51.1,
            files: vec![
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00001-of-00012.safetensors".to_string(),
                    size_bytes: 4_854_573_696,
                    gated: true,
                    sha256: Some("4da0290139f018bdea488b556c136d0f0ca4506fe5f5555cd97c0f6f2e886add"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00002-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_792_944,
                    gated: true,
                    sha256: Some("bf17dbadf9c7cd696e4768639601c3300ea659e49f018000956078cefd475cdf"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00003-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_792_976,
                    gated: true,
                    sha256: Some("c12b9d629d07b4583e19a467713f92b5c1ae8c9d7ef11faf1bdb91c4c7b59efc"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00004-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("7171ed512e46c90cb579a58f66851bad09b028220422ceb1ae85080ab4ffb958"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00005-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("9fb9667695749e55d808f407d78f18f80bd8ff999175c2c480cad7075ff5b2cf"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00006-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("91ae339063266e0c12da89af8aa0cfdb3f9dc9bb1b4b2678863793a28026dbe7"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00007-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("0497dda2b32df5d583caaffae35a96f3524a1e4305b850f4b1ce2e60fc354fe4"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00008-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("9061b71b9cc82e187bd72c8f4594c5c1d900b0bc98c416d72902209514cf8ac4"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00009-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("ebf2e19c3385d4b342e1517639293fe093ad793f41895862713e38603635c769"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00010-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("d651ceb24678d80796a36f9a026f7178631b44e9d86f6f87e52093d915f702ad"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00011-of-00012.safetensors".to_string(),
                    size_bytes: 4_954_793_016,
                    gated: true,
                    sha256: Some("4a2de7fa772158381c7569a5699cadb9da3b06d92b802f54ac1c09f4a2c2e594"),
                },
                ModelFile {
                    component: AiComponent::ModelShard,
                    hf_repo: repo.clone(),
                    hf_filename: "model-00012-of-00012.safetensors".to_string(),
                    size_bytes: 462_476_696,
                    gated: true,
                    sha256: Some("61f4d0c537a889d474396c6fb21ebb90946a64d70345403d47627ecb559e8e91"),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: repo.clone(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 33_384_568,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Config,
                    hf_repo: repo,
                    hf_filename: "config.json".to_string(),
                    size_bytes: 972,
                    gated: true,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image],
    }
}

fn qwen3_4b_q5km_manifest() -> AiModelManifest {
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen3-4b-q5km".to_string(),
            family: "qwen3".to_string(),
            description: "Qwen3 4B Q5_K_M — fast quantized chat & tool calling (~2.9 GB)"
                .to_string(),
            size_gb: 2.9,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: "Qwen/Qwen3-4B-GGUF".to_string(),
                    hf_filename: "Qwen3-4B-Q5_K_M.gguf".to_string(),
                    size_bytes: 2_889_513_184,
                    gated: false,
                    sha256: Some(
                        "aca596860e8cb40af6539e3f2ea40df305f42515deac56d49c08d39a02e6533f",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: "Qwen/Qwen3-4B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool],
    }
}

fn qwen3_30b_a3b_q4km_manifest() -> AiModelManifest {
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen3-30b-a3b-q4km".to_string(),
            family: "qwen3".to_string(),
            description: "Qwen3 30B-A3B Q4_K_M — MoE quantized chat & tool calling (~18.6 GB)"
                .to_string(),
            size_gb: 18.6,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: "Qwen/Qwen3-30B-A3B-GGUF".to_string(),
                    hf_filename: "Qwen3-30B-A3B-Q4_K_M.gguf".to_string(),
                    size_bytes: 18_556_685_824,
                    gated: false,
                    sha256: Some(
                        "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: "Qwen/Qwen3-30B-A3B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool],
    }
}

fn qwen3_8b_q5km_manifest() -> AiModelManifest {
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen3-8b-q5km".to_string(),
            family: "qwen3".to_string(),
            description: "Qwen3 8B Q5_K_M — quantized chat & tool calling (~5.5 GB)".to_string(),
            size_gb: 5.5,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: "Qwen/Qwen3-8B-GGUF".to_string(),
                    hf_filename: "Qwen3-8B-Q5_K_M.gguf".to_string(),
                    size_bytes: 5_851_112_224,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: "Qwen/Qwen3-8B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool],
    }
}

fn qwen3_vl_4b_manifest() -> AiModelManifest {
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen3-vl-4b".to_string(),
            family: "qwen3".to_string(),
            description:
                "Qwen3-VL 4B — quantized multimodal chat, tool & vision (~3.0 GB)".to_string(),
            size_gb: 3.0,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF".to_string(),
                    hf_filename: "Qwen3VL-4B-Instruct-Q4_K_M.gguf".to_string(),
                    size_bytes: 2_497_281_664,
                    gated: false,
                    sha256: Some(
                        "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
                    ),
                },
                ModelFile {
                    component: AiComponent::VisionProjector,
                    hf_repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF".to_string(),
                    hf_filename: "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf".to_string(),
                    size_bytes: 453_974_304,
                    gated: false,
                    sha256: Some(
                        "30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
                    ),
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: "Qwen/Qwen3-VL-4B-Instruct".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 7_032_403,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image],
    }
}

fn qwen3_embed_8b_q5km_manifest() -> AiModelManifest {
    AiModelManifest {
        manifest: ModelManifest {
            name: "qwen3-embed-8b-q5km".to_string(),
            family: "qwen3_embed".to_string(),
            description: "Qwen3 Embedding 8B Q5_K_M — text embeddings (~5.1 GB)".to_string(),
            size_gb: 5.1,
            files: vec![
                ModelFile {
                    component: AiComponent::Model,
                    hf_repo: "Qwen/Qwen3-Embedding-8B-GGUF".to_string(),
                    hf_filename: "Qwen3-Embedding-8B-Q5_K_M.gguf".to_string(),
                    size_bytes: 5_422_342_464,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    component: AiComponent::Tokenizer,
                    hf_repo: "Qwen/Qwen3-Embedding-8B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    size_bytes: 11_422_947,
                    gated: false,
                    sha256: None,
                },
            ],
        },
        categories: vec![ModelCategory::Embed],
    }
}

fn build_all_manifests() -> Vec<AiModelManifest> {
    vec![
        parler_mini_manifest(),
        parler_large_manifest(),
        whisper_large_v3_manifest(),
        whisper_large_v3_turbo_manifest(),
        distil_large_v3_manifest(),
        flux_2_klein_4b_manifest(),
        flux_2_klein_9b_manifest(),
        flux_2_dev_manifest(),
        gemma_3_4b_it_manifest(),
        gemma_3_12b_it_manifest(),
        gemma_3_27b_it_manifest(),
        qwen3_4b_q5km_manifest(),
        qwen3_8b_q5km_manifest(),
        qwen3_30b_a3b_q4km_manifest(),
        qwen3_vl_4b_manifest(),
        qwen3_embed_8b_q5km_manifest(),
    ]
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
    fn known_manifests_contains_parler() {
        let manifests = known_manifests();
        assert!(manifests.len() >= 16);
        assert!(manifests.iter().any(|m| m.manifest.name == "parler-mini"));
        assert!(manifests.iter().any(|m| m.manifest.name == "parler-large"));
    }

    #[test]
    fn known_manifests_contains_whisper() {
        let manifests = known_manifests();
        assert!(manifests.iter().any(|m| m.manifest.name == "whisper-large-v3"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "whisper-large-v3-turbo"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "distil-large-v3"));
    }

    #[test]
    fn whisper_manifests_are_listen_category() {
        for name in ["whisper-large-v3", "whisper-large-v3-turbo", "distil-large-v3"] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "whisper");
            assert!(m.categories.contains(&ModelCategory::Listen));
        }
    }

    #[test]
    fn manifests_for_listen_contains_whisper() {
        let listen = manifests_for_category(ModelCategory::Listen);
        assert!(listen.len() >= 3);
        assert!(listen
            .iter()
            .any(|m| m.manifest.name == "whisper-large-v3-turbo"));
    }

    #[test]
    fn find_manifest_returns_none_for_unknown() {
        assert!(find_manifest("nonexistent-model").is_none());
    }

    #[test]
    fn find_manifest_returns_parler_mini() {
        let m = find_manifest("parler-mini").unwrap();
        assert_eq!(m.manifest.family, "parler");
        assert!(m.categories.contains(&ModelCategory::Talk));
    }

    #[test]
    fn find_manifest_returns_parler_large() {
        let m = find_manifest("parler-large").unwrap();
        assert_eq!(m.manifest.family, "parler");
        assert!(m.categories.contains(&ModelCategory::Talk));
    }

    #[test]
    fn manifests_for_talk_contains_parler() {
        let talk = manifests_for_category(ModelCategory::Talk);
        assert!(talk.len() >= 2);
        assert!(talk.iter().any(|m| m.manifest.name == "parler-mini"));
        assert!(talk.iter().any(|m| m.manifest.name == "parler-large"));
    }

    #[test]
    fn manifests_for_chat_contains_gemma3() {
        let chat = manifests_for_category(ModelCategory::Chat);
        assert!(chat.len() >= 3);
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-3-4b-it"));
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-3-12b-it"));
        assert!(chat.iter().any(|m| m.manifest.name == "gemma-3-27b-it"));
    }

    #[test]
    fn known_manifests_contains_gemma3() {
        let manifests = known_manifests();
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "gemma-3-4b-it"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "gemma-3-12b-it"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "gemma-3-27b-it"));
    }

    #[test]
    fn gemma3_manifests_are_multipurpose() {
        for name in ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "gemma3");
            assert!(m.categories.contains(&ModelCategory::Chat));
            assert!(m.categories.contains(&ModelCategory::Tool));
            assert!(m.categories.contains(&ModelCategory::Image));
        }
    }

    #[test]
    fn gemma3_manifests_are_gated() {
        for name in ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"] {
            let m = find_manifest(name).unwrap();
            assert!(m.manifest.files.iter().all(|f| f.gated));
        }
    }

    #[test]
    fn known_manifests_contains_flux() {
        let manifests = known_manifests();
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "flux-2-klein-4b"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "flux-2-klein-9b"));
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
    fn manifests_for_imagine_contains_flux() {
        let imagine = manifests_for_category(ModelCategory::Imagine);
        assert!(imagine.len() >= 3);
        assert!(imagine
            .iter()
            .any(|m| m.manifest.name == "flux-2-klein-4b"));
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
    fn known_manifests_contains_qwen3() {
        let manifests = known_manifests();
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "qwen3-4b-q5km"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "qwen3-8b-q5km"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "qwen3-30b-a3b-q4km"));
        assert!(manifests
            .iter()
            .any(|m| m.manifest.name == "qwen3-vl-4b"));
    }

    #[test]
    fn qwen3_text_models_are_chat_and_tool() {
        for name in ["qwen3-4b-q5km", "qwen3-8b-q5km", "qwen3-30b-a3b-q4km"] {
            let m = find_manifest(name).unwrap();
            assert_eq!(m.manifest.family, "qwen3");
            assert!(m.categories.contains(&ModelCategory::Chat));
            assert!(m.categories.contains(&ModelCategory::Tool));
            assert!(!m.categories.contains(&ModelCategory::Image));
        }
    }

    #[test]
    fn qwen3_vl_is_multipurpose() {
        let m = find_manifest("qwen3-vl-4b").unwrap();
        assert_eq!(m.manifest.family, "qwen3");
        assert!(m.categories.contains(&ModelCategory::Chat));
        assert!(m.categories.contains(&ModelCategory::Tool));
        assert!(m.categories.contains(&ModelCategory::Image));
    }

    #[test]
    fn qwen3_models_are_ungated() {
        for name in ["qwen3-4b-q5km", "qwen3-8b-q5km", "qwen3-30b-a3b-q4km", "qwen3-vl-4b"] {
            let m = find_manifest(name).unwrap();
            assert!(m.manifest.files.iter().all(|f| !f.gated));
        }
    }

    #[test]
    fn qwen3_vl_has_vision_projector() {
        let m = find_manifest("qwen3-vl-4b").unwrap();
        assert!(m
            .manifest
            .files
            .iter()
            .any(|f| f.component == AiComponent::VisionProjector));
    }

    #[test]
    fn qwen3_embed_is_embed_category() {
        let m = find_manifest("qwen3-embed-8b-q5km").unwrap();
        assert_eq!(m.manifest.family, "qwen3_embed");
        assert_eq!(m.categories, vec![ModelCategory::Embed]);
    }
}
