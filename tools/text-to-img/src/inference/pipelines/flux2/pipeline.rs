//! Flux.2 Klein-4B inference engine.
//!
//! Key differences from FLUX.1:
//! - Uses Qwen3 text encoder (not T5 + CLIP)
//! - Qwen3 hidden states from layers 9, 18, 27 are stacked to produce joint_attention_dim=7680
//! - VAE has latent_channels=32 (not 16)
//! - Transformer has 128 input channels (not 64)
//! - 4D RoPE (not 3D)
//! - Klein is distilled (no guidance embedding)
//! - Linear timestep schedule (distilled, no time-shifting)

use anyhow::{bail, Result};
use local_inference_helpers::candle_core::{DType, Device, IndexOp, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use std::time::Instant;

use super::sampling::{self, Flux2State};
use super::transformer::{Flux2Config, Flux2TransformerWrapper};
use super::vae::{Flux2AutoEncoder, Flux2VaeConfig};
use crate::config::ImageModelPaths;
use crate::inference::encoders;
use crate::inference::image::encode_image;
use crate::inference::{GenerateRequest, GenerateResponse, ImageData, InferenceEngine, LoadStrategy};
use local_inference_helpers::device::{
    free_vram_bytes, preflight_memory_check,
};
use local_inference_helpers::progress::{ProgressCallback, ProgressReporter};

struct LoadedFlux2 {
    transformer: Flux2TransformerWrapper,
    text_encoder: encoders::qwen3::Qwen3Encoder,
    vae: Flux2AutoEncoder,
    device: Device,
    dtype: DType,
}

pub struct Flux2Engine {
    loaded: Option<LoadedFlux2>,
    model_name: String,
    paths: ImageModelPaths,
    progress: ProgressReporter,
    qwen3_variant: Option<String>,
    load_strategy: LoadStrategy,
}

impl Flux2Engine {
    const QWEN3_HIDDEN_LAYERS: [usize; 3] = [9, 18, 27];

    pub fn new(
        model_name: String,
        paths: ImageModelPaths,
        qwen3_variant: Option<String>,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            progress: ProgressReporter::default(),
            qwen3_variant,
            load_strategy,
        }
    }

    fn text_encoder_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.paths.text_encoder_files.is_empty() {
            self.paths.text_encoder_files.clone()
        } else {
            self.paths
                .t5_encoder
                .as_ref()
                .map(|p| vec![p.clone()])
                .unwrap_or_default()
        }
    }

    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path = self
            .paths
            .text_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("text tokenizer path required for Flux.2 models"))?;
        if !text_tokenizer_path.exists() {
            bail!("text tokenizer file not found: {}", text_tokenizer_path.display());
        }
        let encoder_paths = self.text_encoder_paths();
        if encoder_paths.is_empty() {
            bail!("text encoder paths required for Flux.2 models");
        }
        for path in &encoder_paths {
            if !path.exists() {
                bail!("text encoder file not found: {}", path.display());
            }
        }
        if !self.paths.transformer.exists() {
            bail!("transformer file not found: {}", self.paths.transformer.display());
        }
        if !self.paths.vae.exists() {
            bail!("VAE file not found: {}", self.paths.vae.display());
        }
        Ok(text_tokenizer_path.clone())
    }

    fn encode_and_stack(
        encoder: &mut encoders::qwen3::Qwen3Encoder,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let (stacked, _token_count) = encoder.encode_with_layers(
            prompt,
            target_device,
            target_dtype,
            &Self::QWEN3_HIDDEN_LAYERS,
        )?;
        Ok(stacked)
    }

    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;

        let device = local_inference_helpers::device::create_device(|msg| self.progress.info(msg))?;
        let gpu_dtype = local_inference_helpers::dtype::gpu_dtype(&device);

        let start = Instant::now();
        let seed = req.seed;
        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(prompt = %req.prompt, seed, width, height, steps = req.steps, "starting sequential Flux.2 generation");
        self.progress.info("Using sequential loading (load-use-drop) to minimize peak memory");

        // Phase 1: Qwen3 text encoding
        let free = free_vram_bytes().unwrap_or(0);
        self.progress.stage_start("Selecting Qwen3 encoder");
        let resolve_start = Instant::now();
        let (encoder_paths, is_gguf, on_gpu, device_label) = {
            let bf16_paths = self.text_encoder_paths();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            encoders::variant_resolution::resolve_qwen3_variant(
                &self.progress,
                self.qwen3_variant.as_deref(),
                &device,
                free,
                &bf16_paths,
                have_bf16,
                true,
            )?
        };
        self.progress.stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

        let enc_device = if on_gpu { &device } else { &Device::Cpu };
        let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };

        let enc_size: u64 = encoder_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        preflight_memory_check("Qwen3 encoder", enc_size)?;

        let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
        self.progress.stage_start(&enc_stage_label);
        let enc_stage = Instant::now();
        let mut text_encoder = if is_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(&encoder_paths[0], &text_tokenizer_path, enc_device)?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(&encoder_paths, &text_tokenizer_path, enc_device, enc_dtype)?
        };
        self.progress.stage_done(&enc_stage_label, enc_stage.elapsed());

        self.progress.stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();
        let txt_emb = Self::encode_and_stack(&mut text_encoder, &req.prompt, &device, gpu_dtype)?;
        self.progress.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());

        drop(text_encoder);
        self.progress.info("Freed Qwen3 encoder");

        // Phase 2: Load transformer + VAE, denoise
        let flux2_cfg = Flux2Config::klein();

        self.progress.stage_start("Loading Flux.2 transformer (GPU, BF16)");
        let xformer_stage = Instant::now();
        let xformer_paths = if !self.paths.transformer_shards.is_empty() {
            self.paths.transformer_shards.clone()
        } else {
            vec![self.paths.transformer.clone()]
        };
        let flux_vb = unsafe { VarBuilder::from_mmaped_safetensors(&xformer_paths, gpu_dtype, &device)? };
        let transformer = Flux2TransformerWrapper::BF16(super::transformer::Flux2Transformer::new(&flux2_cfg, flux_vb)?);
        self.progress.stage_done("Loading Flux.2 transformer (GPU, BF16)", xformer_stage.elapsed());

        self.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_cfg = Flux2VaeConfig::klein();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&self.paths.vae), gpu_dtype, &device)?
        };
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.progress.stage_done("Loading VAE (GPU)", vae_stage.elapsed());

        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);
        let img = local_inference_helpers::noise::seeded_randn(seed, &[1, 32, latent_h, latent_w], &device, gpu_dtype)?;
        let state = Flux2State::new(&txt_emb, &img)?;

        let image_seq_len = (height / 16) * (width / 16);
        let timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        let denoise_label = format!("Denoising ({} steps)", timesteps.len() - 1);
        self.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let img = transformer.denoise(
            &state.img, &state.img_ids, &state.txt, &state.txt_ids, &state.vec,
            &timesteps, req.guidance, &self.progress,
        )?;
        let img = sampling::unpack(&img, height, width)?;
        self.progress.stage_done(&denoise_label, denoise_start.elapsed());

        drop(transformer);
        drop(state);
        drop(txt_emb);
        device.synchronize()?;

        // Phase 3: VAE decode
        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = vae.decode(&img.to_dtype(gpu_dtype)?)?;
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;
        self.progress.stage_done("VAE decode", vae_decode_start.elapsed());

        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;
        let generation_time_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: self.model_name.clone(),
            seed_used: seed,
        })
    }
}

impl InferenceEngine for Flux2Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        let loaded = self.loaded.as_mut().ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed;
        let width = req.width as usize;
        let height = req.height as usize;

        // Reload Qwen3 encoder if it was dropped
        if loaded.text_encoder.model.is_none() {
            self.progress.stage_start("Reloading Qwen3 encoder");
            let reload_start = Instant::now();
            loaded.text_encoder.reload()?;
            self.progress.stage_done("Reloading Qwen3 encoder", reload_start.elapsed());
        }

        self.progress.stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();
        let txt_emb = Self::encode_and_stack(&mut loaded.text_encoder, &req.prompt, &loaded.device, loaded.dtype)?;
        self.progress.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());

        loaded.text_encoder.drop_weights();

        let latent_h = height.div_ceil(8);
        let latent_w = width.div_ceil(8);
        let img = local_inference_helpers::noise::seeded_randn(seed, &[1, 32, latent_h, latent_w], &loaded.device, loaded.dtype)?;
        let state = Flux2State::new(&txt_emb, &img)?;
        let image_seq_len = (height / 16) * (width / 16);
        let timesteps = sampling::get_schedule(req.steps as usize, image_seq_len);

        let denoise_label = format!("Denoising ({} steps)", timesteps.len() - 1);
        self.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let img = loaded.transformer.denoise(
            &state.img, &state.img_ids, &state.txt, &state.txt_ids, &state.vec,
            &timesteps, req.guidance, &self.progress,
        )?;
        let img = sampling::unpack(&img, height, width)?;
        self.progress.stage_done(&denoise_label, denoise_start.elapsed());

        drop(state);
        drop(txt_emb);

        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img = loaded.vae.decode(&img.to_dtype(loaded.dtype)?)?;
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;
        self.progress.stage_done("VAE decode", vae_decode_start.elapsed());

        let image_bytes = encode_image(&img, req.output_format, req.width, req.height)?;
        let generation_time_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: self.model_name.clone(),
            seed_used: seed,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.load_strategy == LoadStrategy::Sequential || self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }
        if self.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        let text_tokenizer_path = self.validate_paths()?;
        let device = local_inference_helpers::device::create_device(|msg| self.progress.info(msg))?;
        let gpu_dtype = local_inference_helpers::dtype::gpu_dtype(&device);

        // Load transformer
        self.progress.stage_start("Loading Flux.2 transformer (GPU, BF16)");
        let xformer_stage = Instant::now();
        let flux2_cfg = Flux2Config::klein();
        let xformer_paths = if !self.paths.transformer_shards.is_empty() {
            self.paths.transformer_shards.clone()
        } else {
            vec![self.paths.transformer.clone()]
        };
        let flux_vb = unsafe { VarBuilder::from_mmaped_safetensors(&xformer_paths, gpu_dtype, &device)? };
        let transformer = Flux2TransformerWrapper::BF16(super::transformer::Flux2Transformer::new(&flux2_cfg, flux_vb)?);
        self.progress.stage_done("Loading Flux.2 transformer (GPU, BF16)", xformer_stage.elapsed());

        // Load VAE
        self.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        let vae_cfg = Flux2VaeConfig::klein();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&self.paths.vae), gpu_dtype, &device)?
        };
        let vae = Flux2AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.progress.stage_done("Loading VAE (GPU)", vae_stage.elapsed());

        // Load Qwen3 text encoder
        let free = free_vram_bytes().unwrap_or(0);
        self.progress.stage_start("Selecting Qwen3 encoder");
        let resolve_start = Instant::now();
        let (encoder_paths, is_gguf, on_gpu, device_label) = {
            let bf16_paths = self.text_encoder_paths();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            encoders::variant_resolution::resolve_qwen3_variant(
                &self.progress, self.qwen3_variant.as_deref(), &device, free, &bf16_paths, have_bf16, true,
            )?
        };
        self.progress.stage_done("Selecting Qwen3 encoder", resolve_start.elapsed());

        let enc_device = if on_gpu { &device } else { &Device::Cpu };
        let enc_dtype = if on_gpu { gpu_dtype } else { DType::F32 };

        let enc_stage_label = format!("Loading Qwen3 encoder ({device_label})");
        self.progress.stage_start(&enc_stage_label);
        let enc_stage = Instant::now();
        let text_encoder = if is_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(&encoder_paths[0], &text_tokenizer_path, enc_device)?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(&encoder_paths, &text_tokenizer_path, enc_device, enc_dtype)?
        };
        self.progress.stage_done(&enc_stage_label, enc_stage.elapsed());

        self.loaded = Some(LoadedFlux2 {
            transformer,
            text_encoder,
            vae,
            device,
            dtype: gpu_dtype,
        });
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
