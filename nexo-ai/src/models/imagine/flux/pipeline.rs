//! Flux.2 inference pipeline.
//!
//! Uses a sequential load-use-drop pattern to minimize peak memory:
//! 1. Load Qwen3 encoder → encode prompt → drop encoder
//! 2. Load transformer + VAE → denoise → decode → return images

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::config::FluxVariant;
use super::sampling::{self, Flux2State};
use super::transformer::Flux2TransformerWrapper;
use super::vae::Flux2AutoEncoder;
use crate::models::shared::encoders::qwen3::Qwen3Encoder;
use crate::shared::types::{GeneratedImage, ImagineRequest, ImagineResponse};

const QWEN3_HIDDEN_LAYERS: [usize; 3] = [9, 18, 27];

pub struct LoadedState {
    pub device: Device,
    pub dtype: DType,
    pub variant: FluxVariant,
    pub model_dir: PathBuf,
}

/// Discover safetensor files in a subdirectory of the model dir.
fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect();
    paths.sort();
    paths
}

/// Find the tokenizer.json file in the model directory.
fn find_tokenizer(model_dir: &Path) -> Result<PathBuf> {
    // Try tokenizer/ subdirectory first, then text_encoder/
    for subdir in &["tokenizer", "text_encoder"] {
        let path = model_dir.join(subdir).join("tokenizer.json");
        if path.exists() {
            return Ok(path);
        }
    }
    // Try root
    let path = model_dir.join("tokenizer.json");
    if path.exists() {
        return Ok(path);
    }
    bail!(
        "tokenizer.json not found in {}. Looked in tokenizer/, text_encoder/, and root.",
        model_dir.display()
    )
}

pub fn load(model_dir: &Path, variant: FluxVariant) -> Result<LoadedState> {
    let device = crate::device::create_device(|msg| tracing::info!("{msg}"))?;
    let dtype = crate::device::gpu_dtype(&device);

    // Verify critical directories exist
    let transformer_dir = model_dir.join("transformer");
    if !transformer_dir.exists() {
        bail!(
            "transformer directory not found at {}",
            transformer_dir.display()
        );
    }

    let vae_dir = model_dir.join("vae");
    if !vae_dir.exists() {
        bail!("VAE directory not found at {}", vae_dir.display());
    }

    let text_encoder_dir = model_dir.join("text_encoder");
    if !text_encoder_dir.exists() {
        bail!(
            "text_encoder directory not found at {}",
            text_encoder_dir.display()
        );
    }

    find_tokenizer(model_dir)?;

    tracing::info!(
        "Flux.2 {:?} model directory validated at {}",
        variant,
        model_dir.display()
    );

    Ok(LoadedState {
        device,
        dtype,
        variant,
        model_dir: model_dir.to_path_buf(),
    })
}

pub fn imagine(state: &mut LoadedState, request: &ImagineRequest) -> Result<ImagineResponse> {
    let start = Instant::now();
    let width = request.width as usize;
    let height = request.height as usize;
    let steps = request.steps;
    let guidance = request.guidance;
    let seed = request.seed;

    tracing::info!(
        prompt = %request.prompt,
        seed, width, height, steps, guidance,
        "starting Flux.2 generation"
    );

    let model_dir = &state.model_dir;
    let device = &state.device;
    let dtype = state.dtype;

    // Phase 1: Qwen3 text encoding
    let text_encoder_dir = model_dir.join("text_encoder");
    let encoder_paths = find_safetensors(&text_encoder_dir);
    if encoder_paths.is_empty() {
        bail!("no safetensor files found in {}", text_encoder_dir.display());
    }

    let tokenizer_path = find_tokenizer(model_dir)?;

    tracing::info!("loading Qwen3 text encoder ({} shards)", encoder_paths.len());
    let enc_start = Instant::now();
    let text_encoder = Qwen3Encoder::load(&encoder_paths, &tokenizer_path, device, dtype)?;
    tracing::info!("Qwen3 encoder loaded in {:?}", enc_start.elapsed());

    tracing::info!("encoding prompt via Qwen3");
    let encode_start = Instant::now();
    let (txt_emb, token_count) =
        text_encoder.encode_with_layers(&request.prompt, device, dtype, &QWEN3_HIDDEN_LAYERS)?;
    tracing::info!(
        "prompt encoded: {} tokens in {:?}",
        token_count,
        encode_start.elapsed()
    );

    // Drop text encoder to free memory before loading transformer
    drop(text_encoder);
    tracing::info!("freed Qwen3 encoder");

    // Phase 2: Load transformer + VAE, denoise
    let transformer_dir = model_dir.join("transformer");
    let transformer_paths = find_safetensors(&transformer_dir);
    if transformer_paths.is_empty() {
        bail!(
            "no safetensor files found in {}",
            transformer_dir.display()
        );
    }

    let flux2_cfg = state.variant.transformer_config(&transformer_paths)?;

    tracing::info!(
        "loading Flux.2 transformer ({} shards, depth={}, single={})",
        transformer_paths.len(),
        flux2_cfg.depth,
        flux2_cfg.depth_single_blocks,
    );
    let xformer_start = Instant::now();
    let path_strs: Vec<&str> = transformer_paths
        .iter()
        .filter_map(|p| p.to_str())
        .collect();
    let flux_vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
    let transformer = Flux2TransformerWrapper::BF16(
        super::transformer::Flux2Transformer::new(&flux2_cfg, flux_vb)?,
    );
    tracing::info!("transformer loaded in {:?}", xformer_start.elapsed());

    let vae_dir = model_dir.join("vae");
    let vae_paths = find_safetensors(&vae_dir);
    if vae_paths.is_empty() {
        bail!("no safetensor files found in {}", vae_dir.display());
    }

    let vae_cfg = state.variant.vae_config()?;

    tracing::info!("loading VAE");
    let vae_start = Instant::now();
    let vae_path_strs: Vec<&str> = vae_paths.iter().filter_map(|p| p.to_str()).collect();
    let vae_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&vae_path_strs, dtype, device)? };
    let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
    tracing::info!("VAE loaded in {:?}", vae_start.elapsed());

    // Generate initial noise
    let latent_h = height.div_ceil(8);
    let latent_w = width.div_ceil(8);
    let img = seeded_randn(seed, &[1, 32, latent_h, latent_w], device, dtype)?;
    let flux_state = Flux2State::new(&txt_emb, &img)?;

    let image_seq_len = (height / 16) * (width / 16);
    let timesteps = sampling::get_schedule(steps as usize, image_seq_len);

    tracing::info!("denoising ({} steps)", timesteps.len() - 1);
    let denoise_start = Instant::now();

    let img = transformer.denoise(
        &flux_state.img,
        &flux_state.img_ids,
        &flux_state.txt,
        &flux_state.txt_ids,
        &flux_state.vec,
        &timesteps,
        guidance,
    )?;
    let img = sampling::unpack(&img, height, width)?;
    tracing::info!("denoising complete in {:?}", denoise_start.elapsed());

    // Free transformer and state before VAE decode
    drop(transformer);
    drop(flux_state);
    drop(txt_emb);
    let sync_start = Instant::now();
    device.synchronize()?;
    tracing::info!("device sync in {:?}", sync_start.elapsed());

    // Phase 3: VAE decode
    tracing::info!("VAE decode");
    let vae_decode_start = Instant::now();
    let img = vae.decode(&img.to_dtype(dtype)?)?;
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?;
    tracing::info!("VAE decode complete in {:?}", vae_decode_start.elapsed());

    // Convert tensor [3, H, W] to PNG bytes
    let img_data = img
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let image_buffer =
        crate::vision::ImageBuffer::from_rgb(img_data, request.width, request.height)?;
    let png_bytes = crate::vision::encode_png(&image_buffer)?;

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!("total generation time: {}ms", inference_time_ms);

    Ok(ImagineResponse {
        images: vec![GeneratedImage {
            data: png_bytes,
            width: request.width,
            height: request.height,
            index: 0,
        }],
        seed_used: seed,
        inference_time_ms,
    })
}

/// Generate seeded random noise tensor.
fn seeded_randn(
    seed: u64,
    shape: &[usize],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    let elem_count: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..elem_count).map(|_| rng.sample(StandardNormal)).collect();
    let t = Tensor::from_vec(data, shape, &Device::Cpu)?;
    Ok(t.to_dtype(dtype)?.to_device(device)?)
}
