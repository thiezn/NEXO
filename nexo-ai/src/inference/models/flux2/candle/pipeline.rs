//! Flux.2 Klein inference pipeline.
//!
//! Sequential load pattern: Qwen3 text encoder -> encode -> drop -> transformer + VAE -> denoise -> drop -> VAE decode.
//!
//! Key differences from FLUX.1:
//! - Uses Qwen3 text encoder (not T5 + CLIP)
//! - Qwen3 hidden states from layers 9, 18, 27 are stacked to produce `joint_attention_dim=7680`
//! - VAE has `latent_channels=32` (not 16)
//! - Transformer has 128 input channels (not 64)
//! - 4D RoPE (not 3D)
//! - Klein is distilled (no guidance embedding)
//! - No pooled text vector input

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};

use super::config::{Flux2Config, FluxVariant};
use super::sampling::{self, Flux2State};
use super::transformer::Flux2TransformerWrapper;
use super::vae::Flux2AutoEncoder;
use crate::api::types::{GeneratedImage, ImagineRequest, ImagineResponse};

/// Qwen3 hidden layers to extract for Flux.2 text conditioning.
/// Layers 9, 18, 27 correspond to roughly 1/4, 1/2, 3/4 depth of the 36-layer Qwen3.
const QWEN3_HIDDEN_LAYERS: [usize; 3] = [9, 18, 27];

// ---------------------------------------------------------------------------
// Loaded state
// ---------------------------------------------------------------------------

/// Loaded Flux.2 model components, ready for inference.
pub struct LoadedState {
    variant: FluxVariant,
    model_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

/// Find the tokenizer file in the model directory.
fn find_tokenizer(model_dir: &Path) -> Result<PathBuf> {
    let p = model_dir.join("tokenizer.json");
    if p.exists() {
        return Ok(p);
    }
    // Try tokenizer subdirectory
    let p = model_dir.join("tokenizer").join("tokenizer.json");
    if p.exists() {
        return Ok(p);
    }
    // Try text_encoder subdirectory
    let p = model_dir.join("text_encoder").join("tokenizer.json");
    if p.exists() {
        return Ok(p);
    }
    // Try shared flux2 directory
    let shared = crate::download::paths::default_models_dir()
        .join("shared")
        .join("flux2")
        .join("tokenizer")
        .join("tokenizer.json");
    if shared.exists() {
        return Ok(shared);
    }
    bail!("tokenizer.json not found in {}", model_dir.display())
}

/// Find the transformer weights (GGUF or safetensors).
fn find_transformer(model_dir: &Path) -> Result<(Vec<PathBuf>, bool)> {
    // Check for GGUF first
    let gguf_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "gguf"))
        .collect();
    if gguf_files.len() == 1 {
        return Ok((gguf_files, true));
    }

    // Try transformer subdirectory for safetensors
    let xformer_dir = model_dir.join("transformer");
    if xformer_dir.is_dir() {
        let files = crate::inference::candle::weights::find_safetensor_files(&xformer_dir)?;
        return Ok((files, false));
    }

    // Try top-level safetensors matching "transformer" or "diffusion_model"
    let mut files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|f| f.to_str()).is_some_and(|name| {
                name.ends_with(".safetensors")
                    && (name.contains("transformer") || name.contains("diffusion_model"))
            })
        })
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no transformer weights found in {}", model_dir.display());
    }
    Ok((files, false))
}

/// Find the VAE weights.
fn find_vae(model_dir: &Path) -> Result<PathBuf> {
    let vae_dir = model_dir.join("vae");
    if vae_dir.is_dir() {
        let files = crate::inference::candle::weights::find_safetensor_files(&vae_dir)?;
        return Ok(files[0].clone());
    }

    // Try top-level safetensors with "vae" in the name
    let mut files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|name| name.ends_with(".safetensors") && name.contains("vae"))
        })
        .collect();
    files.sort();

    match files.first() {
        Some(f) => Ok(f.clone()),
        None => bail!("no VAE weights found in {}", model_dir.display()),
    }
}

/// Find the text encoder weights.
fn find_text_encoder(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let enc_dir = model_dir.join("text_encoder");
    if enc_dir.is_dir() {
        return crate::inference::candle::weights::find_safetensor_files(&enc_dir);
    }

    // Try top-level safetensors with "text_encoder" in the name
    let mut files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|name| name.ends_with(".safetensors") && name.contains("text_encoder"))
        })
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no text encoder weights found in {}", model_dir.display());
    }
    Ok(files)
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load the model directory and validate that all required files exist.
///
/// The actual weight loading happens lazily in `imagine()` using sequential
/// load-use-drop to minimize peak memory.
pub fn load(model_dir: &Path, variant: FluxVariant) -> Result<LoadedState> {
    // Validate all required files exist
    find_tokenizer(model_dir)?;
    find_transformer(model_dir)?;
    find_vae(model_dir)?;
    find_text_encoder(model_dir)?;

    tracing::info!(
        dir = %model_dir.display(),
        variant = ?variant,
        "Flux.2 model directory validated"
    );

    Ok(LoadedState {
        variant,
        model_dir: model_dir.to_path_buf(),
    })
}

// ---------------------------------------------------------------------------
// Imagine
// ---------------------------------------------------------------------------

/// Generate an image using sequential loading (load-use-drop each component).
pub fn imagine(state: &LoadedState, request: &ImagineRequest) -> Result<ImagineResponse> {
    let start = Instant::now();
    let seed = request.seed;
    let width = request.width as usize;
    let height = request.height as usize;

    tracing::info!(
        prompt = %request.prompt,
        seed, width, height,
        steps = request.steps,
        guidance = request.guidance,
        "starting sequential Flux.2 generation"
    );

    let device = crate::inference::candle::device::create_device()?;
    let gpu_dtype = crate::inference::candle::device::gpu_dtype(&device);

    // --- Phase 1: Qwen3 text encoding ---
    tracing::info!("Phase 1: Loading Qwen3 text encoder");
    let phase1_start = Instant::now();

    let tokenizer_path = find_tokenizer(&state.model_dir)?;
    let encoder_paths = find_text_encoder(&state.model_dir)?;

    let enc_device = &device;
    let enc_dtype = gpu_dtype;

    let encoder = crate::inference::models::support::encoders::qwen3::Qwen3Encoder::load(
        &encoder_paths,
        &tokenizer_path,
        enc_device,
        enc_dtype,
    )?;

    let (txt_emb, token_count) =
        encoder.encode_with_layers(&request.prompt, &device, gpu_dtype, &QWEN3_HIDDEN_LAYERS)?;

    tracing::info!(
        tokens = token_count,
        elapsed_ms = phase1_start.elapsed().as_millis(),
        "Qwen3 encoding complete"
    );

    // Drop text encoder to free memory
    drop(encoder);
    tracing::info!("Qwen3 encoder dropped (sequential mode)");

    // --- Phase 2: Load transformer + VAE, denoise ---
    tracing::info!("Phase 2: Loading transformer + VAE");
    let phase2_start = Instant::now();

    let flux2_cfg = state.variant.transformer_config();
    let (xformer_paths, is_gguf) = find_transformer(&state.model_dir)?;

    let transformer = load_transformer(&flux2_cfg, &xformer_paths, is_gguf, gpu_dtype, &device)?;

    tracing::info!(
        gguf = is_gguf,
        elapsed_ms = phase2_start.elapsed().as_millis(),
        "transformer loaded"
    );

    // Load VAE
    let vae_path = find_vae(&state.model_dir)?;
    let vae_cfg = state.variant.vae_config();
    let vae_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[vae_path.to_str().unwrap_or_default()],
            gpu_dtype,
            &device,
        )?
    };
    let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
    tracing::info!("VAE loaded");

    // Generate noise with seed for reproducibility
    let latent_h = height.div_ceil(8);
    let latent_w = width.div_ceil(8);
    let img = seeded_randn(seed, &[1, 32, latent_h, latent_w], &device, gpu_dtype)?;
    let sampling_state = Flux2State::new(&txt_emb, &img)?;

    // Flux.2 empirical mu schedule (resolution + step-count dependent)
    let image_seq_len = (height / 16) * (width / 16);
    let timesteps = sampling::get_schedule(request.steps as usize, image_seq_len);

    tracing::info!(steps = timesteps.len() - 1, "running denoising loop");
    let denoise_start = Instant::now();

    let img = transformer.denoise(
        &sampling_state.img,
        &sampling_state.img_ids,
        &sampling_state.txt,
        &sampling_state.txt_ids,
        &sampling_state.vec,
        &timesteps,
        request.guidance,
    )?;

    let img = sampling::unpack(&img, height, width)?;

    tracing::info!(
        elapsed_ms = denoise_start.elapsed().as_millis(),
        "denoising complete"
    );

    // Drop transformer + state to free memory for VAE decode
    drop(transformer);
    drop(sampling_state);
    drop(txt_emb);
    device.synchronize()?;
    tracing::info!("transformer dropped, decoding VAE");

    // --- Phase 3: VAE decode ---
    let vae_start = Instant::now();
    let img = vae.decode(&img.to_dtype(gpu_dtype)?)?;

    // Convert to u8 image: clamp to [-1, 1], map to [0, 255]
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?; // remove batch dim: [3, H, W]

    tracing::info!(
        elapsed_ms = vae_start.elapsed().as_millis(),
        "VAE decode complete"
    );

    // Convert candle tensor [3, H, W] -> PNG bytes
    let image_bytes = tensor_to_png(&img, request.width, request.height)?;

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(inference_time_ms, seed, "generation complete");

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load the transformer from either GGUF or safetensors.
fn load_transformer(
    cfg: &Flux2Config,
    paths: &[PathBuf],
    is_gguf: bool,
    gpu_dtype: DType,
    device: &Device,
) -> Result<Flux2TransformerWrapper> {
    if is_gguf {
        let gguf_vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&paths[0], device)?;
        Ok(Flux2TransformerWrapper::Quantized(
            super::quantized_transformer::QuantizedFlux2Transformer::new(
                cfg, gguf_vb, gpu_dtype, device,
            )?,
        ))
    } else {
        let path_strs: Vec<&str> = paths.iter().filter_map(|p| p.to_str()).collect();
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&path_strs, gpu_dtype, device)?
        };
        Ok(Flux2TransformerWrapper::BF16(
            super::transformer::Flux2Transformer::new(cfg, vb)?,
        ))
    }
}

/// Generate reproducible noise on CPU, then move to the target device.
fn seeded_randn(seed: u64, shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0f32, 1.0)?;
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let tensor = Tensor::from_vec(data, shape, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?;
    Ok(tensor)
}

/// Convert a candle tensor [3, H, W] (u8) to PNG bytes.
fn tensor_to_png(img: &Tensor, width: u32, height: u32) -> Result<Vec<u8>> {
    // img shape: [3, H, W], dtype: U8
    // Transpose to [H, W, 3] for image encoding
    let img = img.permute((1, 2, 0))?.contiguous()?;
    let data: Vec<u8> = img.flatten_all()?.to_vec1()?;

    let buffer = crate::vision::ImageBuffer::from_rgb(data, width, height)?;
    crate::vision::encode_png(&buffer)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn seeded_randn_is_reproducible() {
        let a = seeded_randn(42, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(42, &[2, 3], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-10, "same seed should produce identical noise");
    }

    #[test]
    fn seeded_randn_different_seeds_differ() {
        let a = seeded_randn(42, &[100], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(43, &[100], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.1, "different seeds should produce different noise");
    }

    #[test]
    fn tensor_to_png_produces_valid_data() {
        let dev = Device::Cpu;
        let img = Tensor::zeros((3, 4, 4), DType::U8, &dev).unwrap();
        let bytes = tensor_to_png(&img, 4, 4).unwrap();
        // PNG magic bytes
        assert!(bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]));
    }
}
