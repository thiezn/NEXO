use local_inference_helpers::manifest::{Component, ManifestDefaults, ModelFile, ModelManifest};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Image-specific model component types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageComponent {
    Transformer,
    TransformerShard,
    Vae,
    T5Encoder,
    ClipEncoder,
    T5Tokenizer,
    ClipTokenizer,
    TextEncoder,
    TextTokenizer,
}

impl Component for ImageComponent {
    fn name(&self) -> &str {
        match self {
            Self::Transformer => "transformer",
            Self::TransformerShard => "transformer_shard",
            Self::Vae => "vae",
            Self::T5Encoder => "t5_encoder",
            Self::ClipEncoder => "clip_encoder",
            Self::T5Tokenizer => "t5_tokenizer",
            Self::ClipTokenizer => "clip_tokenizer",
            Self::TextEncoder => "text_encoder",
            Self::TextTokenizer => "text_tokenizer",
        }
    }

    fn is_model_specific(&self) -> bool {
        matches!(self, Self::Transformer | Self::TransformerShard)
    }
}

// ── Shared component files ───────────────────────────────────────────────────

/// Shared FLUX component files (VAE, T5, CLIP, tokenizers) -- identical across all FLUX models.
fn shared_flux_files() -> Vec<ModelFile<ImageComponent>> {
    vec![
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
            hf_filename: "ae.safetensors".to_string(),
            component: ImageComponent::Vae,
            size_bytes: 335_000_000,
            gated: true,
            sha256: Some("afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ImageComponent::T5Encoder,
            size_bytes: 9_200_000_000,
            gated: false,
            sha256: Some("6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "clip_l.safetensors".to_string(),
            component: ImageComponent::ClipEncoder,
            size_bytes: 246_000_000,
            gated: false,
            sha256: Some("660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        },
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ImageComponent::T5Tokenizer,
            size_bytes: 2_400_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ImageComponent::ClipTokenizer,
            size_bytes: 600_000,
            gated: false,
            sha256: None,
        },
    ]
}

/// Shared Z-Image component files (Qwen3 text encoder, VAE, tokenizer).
fn shared_zimage_files() -> Vec<ModelFile<ImageComponent>> {
    vec![
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00001-of-00003.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 3_960_000_000,
            gated: false,
            sha256: Some("328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00002-of-00003.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 3_990_000_000,
            gated: false,
            sha256: Some("6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00003-of-00003.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 99_600_000,
            gated: false,
            sha256: Some("7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ImageComponent::Vae,
            size_bytes: 168_000_000,
            gated: false,
            sha256: Some("f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ImageComponent::TextTokenizer,
            size_bytes: 11_400_000,
            gated: false,
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        },
    ]
}

/// Shared Flux.2 Klein-4B component files (Qwen3 text encoder, VAE, tokenizer).
fn shared_flux2_files() -> Vec<ModelFile<ImageComponent>> {
    vec![
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "text_encoder/model-00001-of-00002.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 4_970_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "text_encoder/model-00002-of-00002.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 3_080_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ImageComponent::Vae,
            size_bytes: 160_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ImageComponent::TextTokenizer,
            size_bytes: 11_400_000,
            gated: false,
            sha256: None,
        },
    ]
}

/// Shared Qwen-Image component files (VAE, text encoder shards, tokenizer).
fn shared_qwen_image_files() -> Vec<ModelFile<ImageComponent>> {
    vec![
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ImageComponent::Vae,
            size_bytes: 300_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 4_900_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 4_700_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 4_700_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ImageComponent::TextEncoder,
            size_bytes: 1_200_000_000,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ImageComponent::TextTokenizer,
            size_bytes: 7_000_000,
            gated: false,
            sha256: None,
        },
    ]
}

// ── Manifest registry ────────────────────────────────────────────────────────

static KNOWN_MANIFESTS: LazyLock<Vec<ModelManifest<ImageComponent>>> =
    LazyLock::new(build_known_manifests);

static MANIFEST_INDEX: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    KNOWN_MANIFESTS
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.clone(), i))
        .collect()
});

pub fn known_manifests() -> &'static [ModelManifest<ImageComponent>] {
    &KNOWN_MANIFESTS
}

fn build_known_manifests() -> Vec<ModelManifest<ImageComponent>> {
    let mut manifests = vec![
        // ── FLUX.1 ───────────────────────────────────────────────────────
        ModelManifest {
            name: "flux-schnell:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q8 -- fast 4-step, general purpose".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q8_0.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 12_000_000_000,
                    gated: false,
                    sha256: Some("f6694941193b10148dbf1f0f498d4ccd3e9875c127fc53946213b68580c66f10"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-dev:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q8 -- full quality, 20+ steps".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q8_0.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 12_000_000_000,
                    gated: false,
                    sha256: Some("129032f32224bf7138f16e18673d8008ba5f84c1ec74063bf4511a8bb4cf553d"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-dev:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q4 -- smaller/faster, good quality".to_string(),
            size_gb: 7.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q4_1.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 7_000_000_000,
                    gated: false,
                    sha256: Some("da04c47a9b717bf9a4dd545e46d89e4a62fb44b9497bf9a5d13d622d592fbcda"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-dev:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q6 -- best quality/size trade-off".to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q6_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 9_860_000_000,
                    gated: false,
                    sha256: Some("9566d56031d7f8de184bb5a0393073956ec4e28b32db3f860bd2b87edca04d13"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-schnell:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q4 -- fast 4-step, smaller footprint".to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q4_1.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 7_510_000_000,
                    gated: false,
                    sha256: Some("a798b7196d2fe614cf9bae9a617dbd9f2c14673e454c7f2f6a500347274630b5"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-schnell:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q6 -- fast 4-step, best quality/size trade-off".to_string(),
            size_gb: 9.8,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q6_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 9_830_000_000,
                    gated: false,
                    sha256: Some("a42fd143cec4d7194da281dc8d23a8fe54b16875a13423c042cb545d1da6fa50"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-krea:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q8 -- aesthetic photography fine-tune".to_string(),
            size_gb: 12.7,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q8_0.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 12_700_000_000,
                    gated: false,
                    sha256: Some("0d085b1e3ae0b90e5dbf74da049a80a565617de622a147d28ee37a07761fbd90"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-krea:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q4 -- aesthetic photography, smaller footprint".to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q4_1.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 7_530_000_000,
                    gated: false,
                    sha256: Some("be4c46e5492761f00c0d9ca15e78936fbe54c4ee65b16da8e0dbf5f2115ae6b2"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "flux-krea:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q6 -- aesthetic photography, best quality/size trade-off"
                .to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q6_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 9_860_000_000,
                    gated: false,
                    sha256: Some("c50c13ebe1207b2c87b251ccf3a55b9eb54c84f73cee62503d17acd8a460953e"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
            },
        },
    ];
    manifests.extend(zimage_manifests());
    manifests.extend(flux2_manifests());
    manifests.extend(qwen_image_manifests());
    manifests
}

// ── Z-Image manifests ────────────────────────────────────────────────────────

fn zimage_manifests() -> Vec<ModelManifest<ImageComponent>> {
    vec![
        ModelManifest {
            name: "z-image-turbo:bf16".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo BF16 -- 9-step, Alibaba flow-matching".to_string(),
            size_gb: 24.6,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
                        .to_string(),
                    component: ImageComponent::TransformerShard,
                    size_bytes: 9_970_000_000,
                    gated: false,
                    sha256: Some("95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804"),
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
                        .to_string(),
                    component: ImageComponent::TransformerShard,
                    size_bytes: 9_970_000_000,
                    gated: false,
                    sha256: Some("a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015"),
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
                        .to_string(),
                    component: ImageComponent::TransformerShard,
                    size_bytes: 4_670_000_000,
                    gated: false,
                    sha256: Some("aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "z-image-turbo:q8".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q8 -- 9-step, quantized transformer".to_string(),
            size_gb: 6.58,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q8_0.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 6_580_000_000,
                    gated: false,
                    sha256: Some("df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "z-image-turbo:q6".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q6 -- 9-step, best quality/size trade-off".to_string(),
            size_gb: 5.26,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q6_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 5_260_000_000,
                    gated: false,
                    sha256: Some("319f627beac8059b7546f36a7b4d5097b7f4ee6a1fc37585d0f75ca1d12d01af"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
        ModelManifest {
            name: "z-image-turbo:q4".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q4 -- 9-step, smallest footprint".to_string(),
            size_gb: 3.86,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q4_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 3_860_000_000,
                    gated: false,
                    sha256: Some("14b375ab4f226bc5378f68f37e899ef3c2242b8541e61e2bc1aff40976086fbd"),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
            },
        },
    ]
}

// ── Flux.2 manifests ─────────────────────────────────────────────────────────

fn flux2_manifests() -> Vec<ModelManifest<ImageComponent>> {
    vec![ModelManifest {
        name: "flux2-klein:bf16".to_string(),
        family: "flux2".to_string(),
        description: "[beta] Flux.2 Klein-4B BF16 -- Apache 2.0, 4B param distilled flow-matching"
            .to_string(),
        size_gb: 13.5,
        files: {
            let mut files = shared_flux2_files();
            files.push(ModelFile {
                hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                hf_filename: "transformer/diffusion_pytorch_model.safetensors".to_string(),
                component: ImageComponent::Transformer,
                size_bytes: 7_700_000_000,
                gated: false,
                sha256: None,
            });
            files
        },
        defaults: ManifestDefaults {
            steps: 4,
            guidance: 0.0,
            width: 1024,
            height: 1024,
        },
    }]
}

// ── Qwen-Image manifests ─────────────────────────────────────────────────────

fn qwen_image_manifests() -> Vec<ModelManifest<ImageComponent>> {
    let defaults = ManifestDefaults {
        steps: 30,
        guidance: 0.0,
        width: 1024,
        height: 1024,
    };

    vec![
        ModelManifest {
            name: "qwen-image:bf16".to_string(),
            family: "qwen-image".to_string(),
            description: "[beta] Qwen-Image-2512 BF16 -- 60-block flow-matching transformer"
                .to_string(),
            size_gb: 30.0,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "Qwen/Qwen-Image-2512".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                        .to_string(),
                    component: ImageComponent::TransformerShard,
                    size_bytes: 9_900_000_000,
                    gated: false,
                    sha256: None,
                });
                files.push(ModelFile {
                    hf_repo: "Qwen/Qwen-Image-2512".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                        .to_string(),
                    component: ImageComponent::TransformerShard,
                    size_bytes: 4_700_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        ModelManifest {
            name: "qwen-image:q8".to_string(),
            family: "qwen-image".to_string(),
            description: "[beta] Qwen-Image-2512 Q8 -- quantized transformer, best quality"
                .to_string(),
            size_gb: 21.8,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q8_0.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 21_800_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        ModelManifest {
            name: "qwen-image:q6".to_string(),
            family: "qwen-image".to_string(),
            description: "[beta] Qwen-Image-2512 Q6 -- quantized, best quality/size trade-off"
                .to_string(),
            size_gb: 16.8,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q6_K.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 16_800_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        ModelManifest {
            name: "qwen-image:q4".to_string(),
            family: "qwen-image".to_string(),
            description: "[beta] Qwen-Image-2512 Q4 -- quantized, smallest practical footprint"
                .to_string(),
            size_gb: 12.3,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q4_K_S.gguf".to_string(),
                    component: ImageComponent::Transformer,
                    size_bytes: 12_300_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults,
        },
    ]
}

// ── Text encoder variant registries ──────────────────────────────────────────

/// A quantized T5 encoder variant available from HuggingFace.
#[derive(Debug, Clone)]
pub struct T5Variant {
    pub tag: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
}

/// Known T5 quantized variants, sorted largest to smallest.
pub fn known_t5_variants() -> &'static [T5Variant] {
    static VARIANTS: &[T5Variant] = &[
        T5Variant {
            tag: "q8",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q8_0.gguf",
            size_bytes: 5_060_000_000,
        },
        T5Variant {
            tag: "q6",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q6_K.gguf",
            size_bytes: 3_910_000_000,
        },
        T5Variant {
            tag: "q5",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
            size_bytes: 3_390_000_000,
        },
        T5Variant {
            tag: "q4",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            size_bytes: 2_900_000_000,
        },
        T5Variant {
            tag: "q3",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            size_bytes: 2_100_000_000,
        },
    ];
    VARIANTS
}

pub fn find_t5_variant(tag: &str) -> Option<&'static T5Variant> {
    known_t5_variants().iter().find(|v| v.tag == tag)
}

/// A quantized Qwen3 text encoder variant available from HuggingFace.
#[derive(Debug, Clone)]
pub struct Qwen3Variant {
    pub tag: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
}

/// Known Qwen3 quantized variants, sorted largest to smallest.
pub fn known_qwen3_variants() -> &'static [Qwen3Variant] {
    static VARIANTS: &[Qwen3Variant] = &[
        Qwen3Variant {
            tag: "q8",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-Q8_0.gguf",
            size_bytes: 4_280_000_000,
        },
        Qwen3Variant {
            tag: "q6",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-Q6_K.gguf",
            size_bytes: 3_310_000_000,
        },
        Qwen3Variant {
            tag: "iq4",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-IQ4_XS.gguf",
            size_bytes: 2_270_000_000,
        },
        Qwen3Variant {
            tag: "q3",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-Q3_K_M.gguf",
            size_bytes: 2_080_000_000,
        },
    ];
    VARIANTS
}

pub fn find_qwen3_variant(tag: &str) -> Option<&'static Qwen3Variant> {
    known_qwen3_variants().iter().find(|v| v.tag == tag)
}

// ── Name resolution ──────────────────────────────────────────────────────────

/// Resolve a user-provided model name to its canonical `name:tag` form.
///
/// - `flux-schnell` -> `flux-schnell:q8`
/// - `flux-dev:q4` -> `flux-dev:q4` (unchanged)
/// - `flux-dev-q4` -> `flux-dev:q4` (legacy format)
pub fn resolve_model_name(input: &str) -> String {
    if input.contains(':') {
        return input.to_string();
    }
    // Legacy format: flux-dev-q4 -> flux-dev:q4
    if let Some((base, suffix)) = input.rsplit_once('-')
        && suffix.starts_with('q')
        && suffix.len() <= 3
        && suffix[1..].chars().all(|c| c.is_ascii_digit())
    {
        return format!("{base}:{suffix}");
    }
    // Try :q8 first (FLUX convention), then :bf16 (Z-Image/Flux.2 convention)
    let q8 = format!("{input}:q8");
    if find_manifest_exact(&q8).is_some() {
        return q8;
    }
    let bf16 = format!("{input}:bf16");
    if find_manifest_exact(&bf16).is_some() {
        return bf16;
    }
    format!("{input}:q8")
}

fn find_manifest_exact(name: &str) -> Option<&'static ModelManifest<ImageComponent>> {
    MANIFEST_INDEX.get(name).map(|&i| &KNOWN_MANIFESTS[i])
}

/// Find a manifest by name, handling tag resolution and legacy names.
pub fn find_manifest(name: &str) -> Option<&'static ModelManifest<ImageComponent>> {
    let canonical = resolve_model_name(name);
    MANIFEST_INDEX
        .get(&canonical)
        .map(|&i| &KNOWN_MANIFESTS[i])
}

/// Total size of all files in a manifest in bytes.
pub fn total_download_size(manifest: &ModelManifest<ImageComponent>) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}

/// FP16 T5-XXL model size in bytes (~9.2GB).
pub const T5_FP16_SIZE: u64 = 9_200_000_000;

/// BF16 Qwen3-4B text encoder size in bytes (~8.2GB, 3 safetensors shards).
pub const QWEN3_FP16_SIZE: u64 = 8_200_000_000;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn all_manifests_have_unique_names() {
        let manifests = known_manifests();
        let mut seen = std::collections::HashSet::new();
        for m in manifests {
            assert!(seen.insert(&m.name), "Duplicate manifest name: {}", m.name);
        }
    }

    #[test]
    fn resolve_model_name_with_tag() {
        assert_eq!(resolve_model_name("flux-dev:q4"), "flux-dev:q4");
    }

    #[test]
    fn resolve_model_name_without_tag() {
        assert_eq!(resolve_model_name("flux-schnell"), "flux-schnell:q8");
    }

    #[test]
    fn resolve_model_name_legacy_format() {
        assert_eq!(resolve_model_name("flux-dev-q4"), "flux-dev:q4");
    }

    #[test]
    fn resolve_model_name_bf16_fallback() {
        // flux2-klein only has :bf16, so it falls back
        assert_eq!(resolve_model_name("flux2-klein"), "flux2-klein:bf16");
        // z-image-turbo has :q8, so q8 wins over bf16
        assert_eq!(resolve_model_name("z-image-turbo"), "z-image-turbo:q8");
    }

    #[test]
    fn find_manifest_works() {
        assert!(find_manifest("flux-schnell:q8").is_some());
        assert!(find_manifest("flux-schnell").is_some());
        assert!(find_manifest("z-image-turbo:q8").is_some());
        assert!(find_manifest("flux2-klein:bf16").is_some());
        assert!(find_manifest("qwen-image:bf16").is_some());
        assert!(find_manifest("nonexistent").is_none());
    }

    #[test]
    fn flux_models_have_shared_components() {
        let m = find_manifest("flux-schnell:q8").unwrap();
        assert!(m.files.iter().any(|f| f.component == ImageComponent::Vae));
        assert!(m.files.iter().any(|f| f.component == ImageComponent::T5Encoder));
        assert!(m.files.iter().any(|f| f.component == ImageComponent::ClipEncoder));
    }

    #[test]
    fn total_download_size_nonzero() {
        for m in known_manifests() {
            assert!(total_download_size(m) > 0, "Empty download for {}", m.name);
        }
    }
}
