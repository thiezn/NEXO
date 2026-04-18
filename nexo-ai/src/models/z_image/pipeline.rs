//! Z-Image inference pipeline.
//!
//! Sequential load pattern:
//! 1. Load Qwen3 encoder -> encode prompt -> drop encoder
//! 2. Load transformer + scheduler -> denoise -> drop transformer
//! 3. Load VAE -> decode latents -> encode to PNG
//!
//! This keeps peak memory to max(qwen3, transformer) instead of sum(all).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    AutoEncoderKL, Config, FlowMatchEulerDiscreteScheduler, SchedulerConfig, VaeConfig,
    ZImageTransformer2DModel, calculate_shift, postprocess_image,
};
use candle_transformers::quantized_var_builder;

use super::quantized_transformer::QuantizedZImageTransformer2DModel;
use super::transformer::ZImageTransformer;
use crate::device::{self, preflight_memory_check};
use crate::models::shared::encoders::qwen3::Qwen3Encoder;
use crate::models::shared::weights::{find_gguf_file, find_safetensor_files};
use crate::shared::types::{GeneratedImage, ImagineRequest, ImagineResponse};

/// Z-Image scheduler shift constants (from reference implementation).
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Loaded Z-Image model state (kept between generate calls).
pub struct LoadedState {
    device: Device,
    dtype: DType,
    files: ModelFiles,
}

/// Discover model file layout inside the model directory.
///
/// Expected structure:
/// ```text
/// <model_dir>/
///   transformer/          -- BF16 safetensors shards OR single .gguf
///   vae/                  -- VAE safetensors
///   text_encoder/         -- Qwen3 safetensors shards
///   tokenizer/            -- tokenizer.json
/// ```
struct ModelFiles {
    transformer_paths: Vec<PathBuf>,
    is_gguf: bool,
    vae_path: PathBuf,
    encoder_paths: Vec<PathBuf>,
    tokenizer_path: PathBuf,
}

fn discover_files(model_dir: &Path) -> Result<ModelFiles> {
    let shared_dir = crate::download::paths::default_models_dir()
        .join("shared")
        .join("z_image");

    // Transformer
    let xformer_dir = model_dir.join("transformer");
    if !xformer_dir.exists() {
        bail!("transformer directory not found: {}", xformer_dir.display());
    }

    let (transformer_paths, is_gguf) = if let Ok(gguf) = find_gguf_file(&xformer_dir, "", &[]) {
        (vec![gguf], true)
    } else {
        (find_safetensor_files(&xformer_dir)?, false)
    };

    // VAE — check model dir first, then shared dir
    let vae_dir = model_dir.join("vae");
    let vae_dir = if vae_dir.is_dir() {
        vae_dir
    } else {
        let shared_vae = shared_dir.join("vae");
        if shared_vae.is_dir() {
            shared_vae
        } else {
            bail!(
                "vae directory not found in {} or {}",
                model_dir.display(),
                shared_dir.display()
            );
        }
    };
    let vae_files = find_safetensor_files(&vae_dir)?;
    let vae_path = vae_files
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no VAE safetensors in {}", vae_dir.display()))?;

    // Qwen3 text encoder — check model dir first, then shared dir
    let encoder_dir = model_dir.join("text_encoder");
    let encoder_dir = if encoder_dir.is_dir() {
        encoder_dir
    } else {
        let shared_enc = shared_dir.join("text_encoder");
        if shared_enc.is_dir() {
            shared_enc
        } else {
            bail!(
                "text_encoder directory not found in {} or {}",
                model_dir.display(),
                shared_dir.display()
            );
        }
    };
    let encoder_paths = find_safetensor_files(&encoder_dir)?;

    // Tokenizer — check subdirectory first, then flat
    let tokenizer_path = model_dir.join("tokenizer").join("tokenizer.json");
    let tokenizer_path = if tokenizer_path.exists() {
        tokenizer_path
    } else {
        let flat = model_dir.join("tokenizer.json");
        if flat.exists() {
            flat
        } else {
            bail!("tokenizer.json not found in {}", model_dir.display());
        }
    };

    Ok(ModelFiles {
        transformer_paths,
        is_gguf,
        vae_path,
        encoder_paths,
        tokenizer_path,
    })
}

/// Load the Z-Image model state (device + dtype, files validated).
pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let start = Instant::now();

    let files = discover_files(model_dir)?;
    tracing::info!(
        "Z-Image model files validated in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    let device = device::create_device()?;
    let dtype = device::gpu_dtype(&device);

    Ok(LoadedState {
        device,
        dtype,
        files,
    })
}

/// Generate an image using sequential loading.
pub fn imagine(state: &mut LoadedState, request: &ImagineRequest) -> Result<ImagineResponse> {
    let start = Instant::now();
    let files = &state.files;
    let device = &state.device;
    let dtype = state.dtype;

    let width = request.width as usize;
    let height = request.height as usize;
    let seed = request.seed;
    let num_steps = request.steps as usize;

    tracing::info!(
        prompt = %request.prompt,
        seed, width, height, steps = num_steps,
        "starting Z-Image generation (sequential)"
    );

    // ── Phase 1: Qwen3 text encoding ───────────────────────────────────────

    let encoder_size: u64 = files
        .encoder_paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    preflight_memory_check("Qwen3 text encoder", encoder_size)?;

    tracing::info!(
        "loading Qwen3 text encoder ({} shards)",
        files.encoder_paths.len()
    );
    let encode_start = Instant::now();

    let encoder = Qwen3Encoder::load(&files.encoder_paths, &files.tokenizer_path, device, dtype)?;
    tracing::info!(
        "Qwen3 encoder loaded in {:.1}s",
        encode_start.elapsed().as_secs_f64()
    );

    let prompt_start = Instant::now();
    let (cap_feats, token_count) = encoder.encode(&request.prompt, device, dtype)?;
    let cap_mask = Tensor::ones((1, token_count), DType::U8, device)?;
    tracing::info!(
        token_count,
        "prompt encoded in {:.1}s",
        prompt_start.elapsed().as_secs_f64()
    );

    // Drop encoder to free memory for transformer
    drop(encoder);
    tracing::info!("Qwen3 text encoder dropped (sequential mode)");

    // ── Phase 2: Transformer denoising ─────────────────────────────────────

    let transformer_cfg = Config::z_image_turbo();

    let xformer_size: u64 = files
        .transformer_paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    preflight_memory_check("Z-Image transformer", xformer_size)?;

    tracing::info!(
        quantized = files.is_gguf,
        "loading Z-Image transformer ({} file(s))",
        files.transformer_paths.len()
    );
    let xformer_start = Instant::now();

    let transformer = if files.is_gguf {
        let vb = quantized_var_builder::VarBuilder::from_gguf(&files.transformer_paths[0], device)?;
        ZImageTransformer::Quantized(QuantizedZImageTransformer2DModel::new(
            &transformer_cfg,
            dtype,
            vb,
        )?)
    } else {
        let path_strs: Vec<&str> = files
            .transformer_paths
            .iter()
            .filter_map(|p| p.to_str())
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        ZImageTransformer::BF16(ZImageTransformer2DModel::new(&transformer_cfg, vb)?)
    };

    tracing::info!(
        "transformer loaded in {:.1}s",
        xformer_start.elapsed().as_secs_f64()
    );

    // Calculate latent dimensions: 2 * (image_size / 16)
    let vae_align = 16;
    let latent_h = 2 * (height / vae_align);
    let latent_w = 2 * (width / vae_align);

    // Calculate scheduler shift
    let patch_size = transformer_cfg.all_patch_size[0];
    let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
    let mu = calculate_shift(
        image_seq_len,
        BASE_IMAGE_SEQ_LEN,
        MAX_IMAGE_SEQ_LEN,
        BASE_SHIFT,
        MAX_SHIFT,
    );

    // Initialize scheduler
    let scheduler_cfg = SchedulerConfig::z_image_turbo();
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);
    scheduler.set_timesteps(num_steps, Some(mu));

    // Generate initial noise: (B, 16, latent_h, latent_w) -> unsqueeze -> (B, 16, 1, H, W)
    let mut latents = seeded_randn(seed, &[1, 16, latent_h, latent_w], device, dtype)?;
    latents = latents.unsqueeze(2)?;

    // Denoising loop
    tracing::info!("denoising ({num_steps} steps)");
    let denoise_start = Instant::now();

    for step in 0..num_steps {
        let step_start = Instant::now();
        let t = scheduler.current_timestep_normalized();
        let t_tensor = Tensor::from_vec(vec![t as f32], (1,), device)?.to_dtype(dtype)?;

        // Forward pass through transformer
        let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

        // Negate prediction (Z-Image specific)
        let noise_pred = noise_pred.neg()?;

        // Remove frame dimension for scheduler: (B, C, 1, H, W) -> (B, C, H, W)
        let noise_pred_4d = noise_pred.squeeze(2)?;
        let latents_4d = latents.squeeze(2)?;

        // Scheduler step
        let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;

        // Add back frame dimension
        latents = prev_latents.unsqueeze(2)?;

        tracing::info!(
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

    // Drop transformer and text embeddings to free memory for VAE decode
    drop(transformer);
    drop(cap_feats);
    drop(cap_mask);
    device.synchronize()?;
    tracing::info!("transformer dropped (sequential mode)");

    // ── Phase 3: VAE decode ────────────────────────────────────────────────

    let vae_size = std::fs::metadata(&files.vae_path)
        .map(|m| m.len())
        .unwrap_or(0);
    preflight_memory_check("Z-Image VAE", vae_size)?;

    // On Metal with unified memory, always use GPU for VAE.
    // On CPU-only, use F32.
    let vae_device = device.clone();
    let vae_dtype = if vae_device.is_cpu() {
        DType::F32
    } else {
        dtype
    };

    tracing::info!("loading VAE");
    let vae_load_start = Instant::now();

    let vae_cfg = VaeConfig::z_image();
    let vae_path_str = files
        .vae_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("VAE path is not valid UTF-8"))?;
    let vae_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path_str], vae_dtype, &vae_device)? };
    let vae = AutoEncoderKL::new(&vae_cfg, vae_vb)?;

    tracing::info!(
        "VAE loaded in {:.1}s",
        vae_load_start.elapsed().as_secs_f64()
    );

    tracing::info!("VAE decoding");
    let vae_decode_start = Instant::now();

    // Remove frame dimension: (B, C, 1, H, W) -> (B, C, H, W)
    let latents = latents
        .squeeze(2)?
        .to_device(&vae_device)?
        .to_dtype(vae_dtype)?;
    let image = vae.decode(&latents)?;
    let image = postprocess_image(&image)?;
    let image = image.i(0)?; // Remove batch dimension -> [3, H, W]

    tracing::info!(
        "VAE decode in {:.1}s",
        vae_decode_start.elapsed().as_secs_f64()
    );

    // Encode to PNG
    let image_bytes = tensor_to_png(&image, request.width, request.height)?;

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(inference_time_ms, seed, "Z-Image generation complete");

    Ok(ImagineResponse {
        images: vec![GeneratedImage {
            data: image_bytes,
            width: request.width,
            height: request.height,
            index: 0,
        }],
        seed_used: seed,
        inference_time_ms,
    })
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Generate deterministic initial noise on CPU, then move to target device.
///
/// Using CPU-based RNG ensures identical noise for the same seed across
/// Metal, CUDA, and CPU backends.
fn seeded_randn(seed: u64, shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let elem_count: usize = shape.iter().product();
    let data: Vec<f32> = (0..elem_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let tensor = Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(dtype)?;
    Ok(tensor.to_device(device)?)
}

/// Convert a [3, H, W] U8 tensor to PNG bytes.
fn tensor_to_png(tensor: &Tensor, width: u32, height: u32) -> Result<Vec<u8>> {
    // tensor is [3, H, W] U8
    let tensor = tensor.to_device(&Device::Cpu)?;
    let (c, h, w) = tensor.dims3()?;
    assert_eq!(c, 3);

    // Transpose to [H, W, 3] for image crate
    let data: Vec<u8> = tensor.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;

    let img = image::RgbImage::from_raw(w as u32, h as u32, data)
        .ok_or_else(|| anyhow::anyhow!("failed to create image buffer"))?;

    // Resize if dimensions don't match requested size
    let img = if w as u32 != width || h as u32 != height {
        image::DynamicImage::ImageRgb8(img).resize_exact(
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        image::DynamicImage::ImageRgb8(img)
    };

    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(buf.into_inner())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn latent_dimensions() {
        // 1024px -> 2 * (1024 / 16) = 128
        assert_eq!(2 * (1024 / 16), 128);
        // 512px -> 2 * (512 / 16) = 64
        assert_eq!(2 * (512 / 16), 64);
        // 768px -> 2 * (768 / 16) = 96
        assert_eq!(2 * (768 / 16), 96);
    }

    #[test]
    fn scheduler_shift_calculation() {
        let mu = calculate_shift(
            4096,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        // At max seq len, mu should be max_shift
        assert!((mu - MAX_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn scheduler_shift_at_base() {
        let mu = calculate_shift(
            BASE_IMAGE_SEQ_LEN,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        // At base seq len, mu should be base_shift
        assert!((mu - BASE_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn scheduler_shift_at_1024x1024() {
        // 1024x1024 -> latent 128x128 -> patch 2 -> 64*64 = 4096 seq len
        let latent_h = 2 * (1024 / 16);
        let latent_w = 2 * (1024 / 16);
        let patch_size = 2;
        let seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        assert_eq!(seq_len, 4096);

        let mu = calculate_shift(
            seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!((mu - MAX_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn seeded_randn_deterministic() {
        let a = seeded_randn(42, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(42, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn seeded_randn_different_seeds() {
        let a = seeded_randn(42, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(43, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0);
    }

    #[test]
    fn tensor_to_png_roundtrip() {
        // Create a 4x4 RGB tensor
        let data: Vec<u8> = (0..48).collect();
        let tensor = Tensor::from_vec(data, (3, 4, 4), &Device::Cpu).unwrap();
        let png = tensor_to_png(&tensor, 4, 4).unwrap();
        assert!(!png.is_empty());
        // PNG magic bytes
        assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
