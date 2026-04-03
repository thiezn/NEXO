//! Qwen-Image-2512 inference pipeline.
//!
//! Pipeline: Qwen2.5-VL text encoder -> QwenImageTransformer2DModel -> QwenImage VAE
//!
//! Architecture follows Z-Image closely (both from Alibaba/Tongyi):
//! - Dual-stream transformer with joint attention and 3D RoPE
//! - Flow-matching Euler discrete scheduler with dynamic shifting
//! - Sequential loading strategy: load text encoder -> encode -> drop -> load
//!   transformer + VAE -> denoise -> decode
//!
//! Key differences from Z-Image:
//! - 60 identical dual-stream blocks (no noise_refiner/context_refiner)
//! - Qwen2.5-VL text encoder (hidden_size=3584) instead of Qwen3 (2560)
//! - Custom VAE with per-channel latent normalization
//! - ComfyUI-style SNR time shift scheduling (shift=3.1)

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::z_image::postprocess_image;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::quantized_transformer::QuantizedQwenImageTransformer2DModel;
use super::sampling::{DEFAULT_SHIFT, QwenImageScheduler};
use super::transformer::{QwenImageConfig, QwenImageTransformer2DModel};
use super::vae::QwenImageVae;
use crate::models::shared::encoders::qwen2_text::Qwen2TextEncoder;
use crate::shared::types::{GeneratedImage, ImagineRequest, ImagineResponse};

/// Minimum free memory (bytes) required to place Qwen-Image VAE on GPU.
const VAE_DECODE_MEMORY_THRESHOLD: u64 = 2_500_000_000;

#[allow(clippy::large_enum_variant)]
enum QwenImageTransformer {
    BF16(QwenImageTransformer2DModel),
    Quantized(QuantizedQwenImageTransformer2DModel),
}

impl QwenImageTransformer {
    fn forward(
        &self,
        latents: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
            Self::Quantized(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
        }
    }
}

/// Loaded state for Qwen-Image pipeline.
pub(crate) struct LoadedState {
    transformer: QwenImageTransformer,
    vae: QwenImageVae,
}

/// Generate a seeded random normal tensor on CPU for reproducible results.
fn seeded_randn(seed: u64, shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = StdRng::seed_from_u64(seed);
    let elem_count: usize = shape.iter().product();
    let data: Vec<f32> = (0..elem_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let tensor = Tensor::from_vec(data, shape, &Device::Cpu)?;
    Ok(tensor.to_device(device)?.to_dtype(dtype)?)
}

/// Discover model files in the model directory.
struct ModelFiles {
    /// Transformer safetensors shards or single GGUF file
    transformer_paths: Vec<PathBuf>,
    /// VAE safetensors path
    vae_path: PathBuf,
    /// Text encoder safetensors shards
    text_encoder_paths: Vec<PathBuf>,
    /// Text tokenizer path
    tokenizer_path: PathBuf,
    /// Whether the transformer is GGUF quantized
    is_quantized: bool,
}

impl ModelFiles {
    fn discover(model_dir: &Path) -> Result<Self> {
        // Look for transformer files
        let transformer_dir = model_dir.join("transformer");
        let (transformer_paths, is_quantized) = if transformer_dir.exists() {
            // BF16 safetensors shards in transformer/ subdirectory
            let paths = crate::models::shared::weights::find_safetensor_files(&transformer_dir)?;
            (paths, false)
        } else {
            // Look for GGUF file in model root
            let gguf = crate::models::shared::weights::find_gguf_file(
                model_dir,
                "",
                &["vae", "text_encoder"],
            );
            match gguf {
                Ok(path) => (vec![path], true),
                Err(_) => {
                    // Try safetensors in root
                    let paths = crate::models::shared::weights::find_safetensor_files(model_dir)?;
                    (paths, false)
                }
            }
        };

        // VAE
        let vae_dir = model_dir.join("vae");
        let vae_path = if vae_dir.exists() {
            let files = crate::models::shared::weights::find_safetensor_files(&vae_dir)?;
            files
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("no VAE safetensors found"))?
        } else {
            model_dir.join("vae.safetensors")
        };
        if !vae_path.exists() {
            bail!("VAE file not found: {}", vae_path.display());
        }

        // Text encoder
        let te_dir = model_dir.join("text_encoder");
        let text_encoder_paths = if te_dir.exists() {
            crate::models::shared::weights::find_safetensor_files(&te_dir)?
        } else {
            bail!(
                "text_encoder directory not found in {}",
                model_dir.display()
            );
        };

        // Tokenizer
        let tokenizer_path = te_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            // Try model root
            let alt = model_dir.join("tokenizer.json");
            if !alt.exists() {
                bail!(
                    "tokenizer.json not found in {} or {}",
                    te_dir.display(),
                    model_dir.display()
                );
            }
            return Ok(Self {
                transformer_paths,
                vae_path,
                text_encoder_paths,
                tokenizer_path: alt,
                is_quantized,
            });
        }

        Ok(Self {
            transformer_paths,
            vae_path,
            text_encoder_paths,
            tokenizer_path,
            is_quantized,
        })
    }
}

/// Load the text encoder, encode the prompt, and drop the encoder.
fn encode_prompt(
    files: &ModelFiles,
    prompt: &str,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    tracing::info!(
        "loading Qwen2.5-VL text encoder ({} shards)",
        files.text_encoder_paths.len()
    );
    let load_start = Instant::now();
    let encoder = Qwen2TextEncoder::load(
        &files.text_encoder_paths,
        &files.tokenizer_path,
        device,
        dtype,
    )?;
    tracing::info!(
        "text encoder loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    let encode_start = Instant::now();
    let (hidden_states, _attention_mask, valid_len) = encoder.encode(prompt, device, dtype)?;
    tracing::info!(
        "prompt encoded in {:.1}s (valid_len={})",
        encode_start.elapsed().as_secs_f64(),
        valid_len
    );

    // Build mask from valid_len
    let mut mask = vec![0u8; hidden_states.dim(1)?];
    for value in &mut mask[..valid_len] {
        *value = 1;
    }
    let attention_mask = Tensor::from_vec(mask, (1, hidden_states.dim(1)?), device)?;

    // Encoder dropped here, freeing memory
    drop(encoder);
    tracing::info!("text encoder dropped");

    Ok((hidden_states, attention_mask))
}

/// Load the transformer from disk.
fn load_transformer(
    files: &ModelFiles,
    device: &Device,
    dtype: DType,
    cfg: &QwenImageConfig,
) -> Result<QwenImageTransformer> {
    if files.is_quantized {
        tracing::info!("loading quantized Qwen-Image transformer (GGUF)");
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &files.transformer_paths[0],
            device,
        )?;
        Ok(QwenImageTransformer::Quantized(
            QuantizedQwenImageTransformer2DModel::new(cfg, vb, device)?,
        ))
    } else {
        tracing::info!(
            "loading BF16 Qwen-Image transformer ({} shards)",
            files.transformer_paths.len()
        );
        let path_strs: Vec<&str> = files
            .transformer_paths
            .iter()
            .filter_map(|p| p.to_str())
            .collect();
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        Ok(QwenImageTransformer::BF16(
            QwenImageTransformer2DModel::new(cfg, vb)?,
        ))
    }
}

/// Load the VAE from disk.
fn load_vae(files: &ModelFiles, device: &Device, dtype: DType) -> Result<QwenImageVae> {
    Ok(QwenImageVae::load(&files.vae_path, device, dtype)?)
}

/// Load all components from the model directory (sequential strategy).
///
/// This uses the sequential loading pattern:
/// 1. Load text encoder -> encode prompt -> drop encoder
/// 2. Load transformer + VAE -> denoise -> decode
pub(crate) fn load(model_dir: &Path) -> Result<LoadedState> {
    let files = ModelFiles::discover(model_dir)?;
    let device = crate::device::create_device()?;

    // Use BF16 on GPU (matches Qwen-Image training dtype), F32 on CPU
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let cfg = QwenImageConfig::qwen_image_2512();

    tracing::info!("loading Qwen-Image transformer");
    let load_start = Instant::now();
    let transformer = load_transformer(&files, &device, dtype, &cfg)?;
    tracing::info!(
        "transformer loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    // VAE always in F32 -- BF16 convolutions accumulate quantization noise
    let vae_dtype = DType::F32;
    let vae_device = resolve_vae_device(&device);
    tracing::info!("loading Qwen-Image VAE");
    let vae_start = Instant::now();
    let vae = load_vae(&files, &vae_device, vae_dtype)?;
    tracing::info!("VAE loaded in {:.1}s", vae_start.elapsed().as_secs_f64());

    Ok(LoadedState { transformer, vae })
}

/// Determine where to place the VAE based on available memory.
fn resolve_vae_device(gpu_device: &Device) -> Device {
    if gpu_device.is_cpu() {
        return Device::Cpu;
    }
    // On macOS with Metal, check available system memory
    let available = crate::device::available_system_memory_bytes().unwrap_or(0);
    if available > VAE_DECODE_MEMORY_THRESHOLD {
        gpu_device.clone()
    } else {
        tracing::info!(
            "insufficient memory for VAE on GPU ({} available), using CPU",
            crate::device::fmt_gb(available)
        );
        Device::Cpu
    }
}

/// Run the full Qwen-Image generation pipeline.
pub(crate) fn generate(
    loaded: &LoadedState,
    model_dir: &Path,
    request: &ImagineRequest,
) -> Result<ImagineResponse> {
    let files = ModelFiles::discover(model_dir)?;
    let device = crate::device::create_device()?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };

    let start = Instant::now();
    let seed = request.seed;
    let width = request.width as usize;
    let height = request.height as usize;

    tracing::info!(
        prompt = %request.prompt,
        seed, width, height,
        steps = request.steps,
        "starting Qwen-Image generation"
    );

    // --- Phase 1: Text encoding (load-use-drop) ---
    let (encoder_hidden_states, encoder_attention_mask) =
        encode_prompt(&files, &request.prompt, &device, dtype)?;

    // Encode unconditional (empty) prompt for classifier-free guidance
    let use_cfg = request.guidance > 1.0;
    let (uncond_hs, uncond_mask) = if use_cfg {
        let (hs, mask) = encode_prompt(&files, "", &device, dtype)?;
        (Some(hs), Some(mask))
    } else {
        (None, None)
    };

    // --- Phase 2: Denoise ---
    let vae_downsample = 8;
    let latent_h = height / vae_downsample;
    let latent_w = width / vae_downsample;
    let num_steps = request.steps as usize;

    let mut scheduler = QwenImageScheduler::new(num_steps, DEFAULT_SHIFT);

    // Initial noise scaled by sigma[0] (ComfyUI CONST noise scaling)
    let mut latents = seeded_randn(seed, &[1, 16, latent_h, latent_w], &device, dtype)?;
    latents = (latents * scheduler.initial_sigma())?;

    tracing::info!("denoising ({num_steps} steps)");
    let denoise_start = Instant::now();

    // Pre-batch CFG inputs to halve block transfers for GGUF streaming.
    let (batched_hs, batched_mask) = if use_cfg {
        let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
        let mask = Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
        (hs, mask)
    } else {
        (
            encoder_hidden_states.clone(),
            encoder_attention_mask.clone(),
        )
    };

    for step in 0..num_steps {
        let step_start = Instant::now();
        let t = scheduler.current_timestep();
        let noise_pred = if use_cfg {
            let t_tensor = Tensor::from_vec(vec![t as f32; 2], (2,), &device)?.to_dtype(dtype)?;
            let batched_latents = Tensor::cat(&[&latents, &latents], 0)?;
            let batched_pred = loaded.transformer.forward(
                &batched_latents,
                &t_tensor,
                &batched_hs,
                &batched_mask,
            )?;
            let cond_pred = batched_pred.narrow(0, 0, 1)?;
            let uncond_pred = batched_pred.narrow(0, 1, 1)?;
            // CFG in F32 to avoid BF16 cancellation error, then norm rescale
            let cond_f32 = cond_pred.to_dtype(DType::F32)?;
            let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
            let comb = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * request.guidance)?)?;
            // Rescale: comb * (norm(cond) / norm(comb)) per-pixel along channels
            let cond_norm = cond_f32.sqr()?.sum_keepdim(1)?.sqrt()?;
            let comb_norm = comb.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-8, f64::MAX)?;
            let rescaled = comb.broadcast_mul(&(cond_norm / comb_norm)?)?;
            rescaled.to_dtype(dtype)?
        } else {
            let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
            loaded.transformer.forward(
                &latents,
                &t_tensor,
                &encoder_hidden_states,
                &encoder_attention_mask,
            )?
        };
        latents = scheduler.step(&noise_pred, &latents)?;
        tracing::debug!(
            step = step + 1,
            total = num_steps,
            elapsed_ms = step_start.elapsed().as_millis(),
            "denoise step"
        );
    }

    tracing::info!(
        "denoising complete in {:.1}s",
        denoise_start.elapsed().as_secs_f64()
    );

    // --- Phase 3: VAE decode ---
    tracing::info!("VAE decode");
    let vae_start = Instant::now();

    // Always decode in F32
    let vae_device = resolve_vae_device(&device);
    let latents = latents.to_device(&vae_device)?.to_dtype(DType::F32)?;
    let image = loaded.vae.decode(&latents)?;
    let image = postprocess_image(&image)?;
    let image = image.i(0)?;

    tracing::info!(
        "VAE decode complete in {:.1}s",
        vae_start.elapsed().as_secs_f64()
    );

    // Encode to PNG
    let (h, w, _c) = image.dims3()?;
    let image_data = image.to_vec3::<u8>()?;
    let mut png_data = Vec::new();
    {
        let encoder = image::codecs::png::PngEncoder::new(&mut png_data);
        use image::ImageEncoder;
        let flat: Vec<u8> = image_data.into_iter().flatten().flatten().collect();
        encoder.write_image(&flat, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    }

    let generation_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(generation_time_ms, seed, "Qwen-Image generation complete");

    Ok(ImagineResponse {
        images: vec![GeneratedImage {
            data: png_data,
            width: request.width,
            height: request.height,
            index: 0,
        }],
        seed_used: seed,
        inference_time_ms: generation_time_ms,
    })
}
