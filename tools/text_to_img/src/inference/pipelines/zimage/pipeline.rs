//! Z-Image Turbo inference engine.

use anyhow::{bail, Result};
use local_inference_helpers::candle_core::{DType, Device, IndexOp, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, postprocess_image, AutoEncoderKL, Config, FlowMatchEulerDiscreteScheduler,
    SchedulerConfig, VaeConfig, ZImageTransformer2DModel,
};
use candle_transformers::quantized_var_builder;
use std::time::Instant;

use super::quantized_transformer::QuantizedZImageTransformer2DModel;
use super::transformer::ZImageTransformer;
use crate::config::ImageModelPaths;
use crate::inference::encoders;
use crate::inference::image::encode_image;
use crate::inference::{
    GenerateRequest, GenerateResponse, ImageData, InferenceEngine, LoadStrategy,
};
use local_inference_helpers::device::{
    fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check, should_use_gpu,
};
use local_inference_helpers::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Z-Image scheduler shift constants (from reference implementation).
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Minimum free VRAM (bytes) required to place Z-Image VAE on GPU.
/// The VAE itself is small (~160MB), but decode at 1024x1024 needs ~6GB workspace
/// for conv2d im2col expansions through the upsampling blocks.
const VAE_DECODE_VRAM_THRESHOLD: u64 = 6_500_000_000;

/// Loaded Z-Image model components, ready for inference.
struct LoadedZImage {
    transformer: Option<ZImageTransformer>,
    text_encoder: encoders::qwen3::Qwen3Encoder,
    vae: AutoEncoderKL,
    transformer_cfg: Config,
    device: Device,
    vae_device: Device,
    dtype: DType,
    is_quantized: bool,
    vae_path: std::path::PathBuf,
}

/// Z-Image inference engine backed by candle's z_image module.
pub struct ZImageEngine {
    loaded: Option<LoadedZImage>,
    model_name: String,
    paths: ImageModelPaths,
    progress: ProgressReporter,
    qwen3_variant: Option<String>,
    load_strategy: LoadStrategy,
}

impl ZImageEngine {
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

    fn transformer_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.paths.transformer_shards.is_empty() {
            self.paths.transformer_shards.clone()
        } else {
            vec![self.paths.transformer.clone()]
        }
    }

    fn detect_is_gguf(&self) -> bool {
        self.paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path =
            self.paths.text_tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("text tokenizer path required for Z-Image models")
            })?;
        if !text_tokenizer_path.exists() {
            bail!(
                "text tokenizer file not found: {}",
                text_tokenizer_path.display()
            );
        }

        let xformer_paths = self.transformer_paths();
        for path in &xformer_paths {
            if !path.exists() {
                bail!("transformer file not found: {}", path.display());
            }
        }
        if !self.paths.vae.exists() {
            bail!("VAE file not found: {}", self.paths.vae.display());
        }

        Ok(text_tokenizer_path.clone())
    }

    fn load_transformer(
        &self,
        device: &Device,
        dtype: DType,
        cfg: &Config,
    ) -> Result<ZImageTransformer> {
        let is_gguf = self.detect_is_gguf();
        let xformer_paths = self.transformer_paths();

        if is_gguf {
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.paths.transformer, device)?;
            Ok(ZImageTransformer::Quantized(
                QuantizedZImageTransformer2DModel::new(cfg, dtype, vb)?,
            ))
        } else {
            let xformer_vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&xformer_paths, dtype, device)? };
            Ok(ZImageTransformer::BF16(ZImageTransformer2DModel::new(
                cfg, xformer_vb,
            )?))
        }
    }

    fn load_vae(&self, device: &Device, dtype: DType) -> Result<AutoEncoderKL> {
        let vae_cfg = VaeConfig::z_image();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[self.paths.vae.as_path()], dtype, device)?
        };
        Ok(AutoEncoderKL::new(&vae_cfg, vae_vb)?)
    }

    fn reload_transformer(&self, loaded: &mut LoadedZImage) -> Result<()> {
        let transformer =
            self.load_transformer(&loaded.device, loaded.dtype, &loaded.transformer_cfg)?;
        loaded.transformer = Some(transformer);
        Ok(())
    }

    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let is_gguf = self.detect_is_gguf();
        let transformer_cfg = Config::z_image_turbo();

        let device =
            local_inference_helpers::device::create_device(|msg| self.progress.info(msg))?;
        let dtype = local_inference_helpers::dtype::gpu_dtype(&device);

        let start = Instant::now();
        let seed = req.seed;
        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Z-Image generation"
        );

        self.progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Qwen3 text encoding ---
        let free = free_vram_bytes().unwrap_or(0);
        self.progress.stage_start("Selecting Qwen3 encoder");
        let qwen3_resolve_start = Instant::now();
        let (resolved_paths, is_qwen3_gguf, te_on_gpu, te_device_label) = {
            let bf16_paths = self.paths.text_encoder_files.clone();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            encoders::variant_resolution::resolve_qwen3_variant(
                &self.progress,
                self.qwen3_variant.as_deref(),
                &device,
                free,
                &bf16_paths,
                have_bf16,
                false,
            )?
        };
        self.progress
            .stage_done("Selecting Qwen3 encoder", qwen3_resolve_start.elapsed());

        let te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

        let te_size: u64 = resolved_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        preflight_memory_check("Qwen3 text encoder", te_size)?;

        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        let te_label = if is_qwen3_gguf {
            format!("Loading Qwen3 text encoder (GGUF, {})", te_device_label)
        } else {
            format!(
                "Loading Qwen3 text encoder ({} shards, {})",
                resolved_paths.len(),
                te_device_label,
            )
        };
        self.progress.stage_start(&te_label);
        let te_start = Instant::now();

        let mut text_encoder = if is_qwen3_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(
                &resolved_paths[0],
                &text_tokenizer_path,
                &te_device,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(
                &resolved_paths,
                &text_tokenizer_path,
                &te_device,
                te_dtype,
            )?
        };
        self.progress.stage_done(&te_label, te_start.elapsed());

        self.progress.stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();
        let (cap_feats, token_count) = text_encoder.encode(&req.prompt, &device, dtype)?;
        let cap_mask = Tensor::ones((1, token_count), DType::U8, &device)?;
        self.progress
            .stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());

        drop(text_encoder);
        self.progress.info("Freed Qwen3 text encoder");

        // --- Phase 2: Load transformer and denoise ---
        let xformer_paths = self.transformer_paths();
        let xformer_size: u64 = xformer_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        preflight_memory_check("Z-Image transformer", xformer_size)?;

        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }

        let xformer_label = if is_gguf {
            "Loading Z-Image transformer (GPU, quantized)".to_string()
        } else {
            format!(
                "Loading Z-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;
        self.progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        // Calculate latent dimensions
        let vae_align = 16;
        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        let patch_size = transformer_cfg.all_patch_size[0];
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );

        let scheduler_cfg = SchedulerConfig::z_image_turbo();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);
        scheduler.set_timesteps(req.steps as usize, Some(mu));

        let mut latents = local_inference_helpers::noise::seeded_randn(
            seed,
            &[1, 16, latent_h, latent_w],
            &device,
            dtype,
        )?;
        latents = latents.unsqueeze(2)?;

        let num_steps = req.steps as usize;
        let denoise_label = format!("Denoising ({num_steps} steps)");
        self.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for step in 0..num_steps {
            let step_start = Instant::now();
            let t = scheduler.current_timestep_normalized();
            let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
            let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;
            let noise_pred = noise_pred.neg()?;
            let noise_pred_4d = noise_pred.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;
            let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;
            latents = prev_latents.unsqueeze(2)?;
            self.progress.emit(ProgressEvent::Step {
                step: step + 1,
                total: num_steps,
                elapsed: step_start.elapsed(),
            });
        }

        self.progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        drop(transformer);
        drop(cap_feats);
        drop(cap_mask);
        device.synchronize()?;
        self.progress.info("Freed Z-Image transformer");

        // --- Phase 3: Load VAE and decode ---
        if let Some(status) = memory_status_string() {
            self.progress.info(&status);
        }
        let free_for_vae = free_vram_bytes().unwrap_or(0);
        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_vae,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let vae_dtype = if vae_on_gpu { dtype } else { DType::F32 };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        let vae_label = format!("Loading VAE ({vae_device_label})");
        self.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.progress.stage_done(&vae_label, vae_start.elapsed());

        self.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let latents = latents
            .squeeze(2)?
            .to_device(&vae_device)?
            .to_dtype(vae_dtype)?;
        let image = vae.decode(&latents)?;
        let image = postprocess_image(&image)?;
        let image = image.i(0)?;

        self.progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        let image_bytes = encode_image(&image, req.output_format, req.width, req.height)?;
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

impl InferenceEngine for ZImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Sequential mode: load-use-drop each component
        if self.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.loaded.is_none() {
            bail!("model not loaded — call load() first");
        }

        let progress = &self.progress;
        let start = Instant::now();

        // Reload transformer if it was dropped after previous VAE decode
        let loaded_ref = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let needs_reload = loaded_ref.transformer.is_none();
        if needs_reload {
            let mut loaded_mut = self
                .loaded
                .take()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
            let xformer_label = if loaded_mut.is_quantized {
                "Reloading Z-Image transformer (GPU, quantized)"
            } else {
                "Reloading Z-Image transformer (GPU, BF16)"
            };
            progress.stage_start(xformer_label);
            let reload_start = Instant::now();
            self.reload_transformer(&mut loaded_mut)?;
            progress.stage_done(xformer_label, reload_start.elapsed());
            self.loaded = Some(loaded_mut);
        }

        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let seed = req.seed;
        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Z-Image generation"
        );

        // 1. Reload text encoder if weights were dropped
        if loaded.text_encoder.model.is_none() {
            let te_label = if loaded.text_encoder.is_quantized {
                "Reloading Qwen3 encoder (GGUF)"
            } else {
                "Reloading Qwen3 encoder (BF16)"
            };
            progress.stage_start(te_label);
            let reload_start = Instant::now();
            loaded.text_encoder.reload()?;
            progress.stage_done(te_label, reload_start.elapsed());
        }

        // 2. Encode prompt with Qwen3
        progress.stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();

        let (cap_feats, token_count) =
            loaded
                .text_encoder
                .encode(&req.prompt, &loaded.device, loaded.dtype)?;
        let cap_mask = Tensor::ones((1, token_count), DType::U8, &loaded.device)?;

        progress.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());

        // Drop text encoder from GPU to free VRAM for denoising + VAE decode
        loaded.text_encoder.drop_weights();

        // 3. Calculate latent dimensions: 2 * (image_size / 16)
        let vae_align = 16;
        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        // 4. Calculate scheduler shift
        let patch_size = loaded.transformer_cfg.all_patch_size[0];
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );

        // 5. Initialize scheduler
        let scheduler_cfg = SchedulerConfig::z_image_turbo();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);
        scheduler.set_timesteps(req.steps as usize, Some(mu));

        // 6. Generate initial noise
        let mut latents = local_inference_helpers::noise::seeded_randn(
            seed,
            &[1, 16, latent_h, latent_w],
            &loaded.device,
            loaded.dtype,
        )?;
        latents = latents.unsqueeze(2)?;

        // 7. Denoising loop
        let num_steps = req.steps as usize;
        let denoise_label = format!("Denoising ({num_steps} steps)");
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            for step in 0..num_steps {
                let step_start = Instant::now();
                let t = scheduler.current_timestep_normalized();
                let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                    .to_dtype(loaded.dtype)?;

                let noise_pred =
                    transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;
                let noise_pred = noise_pred.neg()?;

                let noise_pred_4d = noise_pred.squeeze(2)?;
                let latents_4d = latents.squeeze(2)?;
                let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;
                latents = prev_latents.unsqueeze(2)?;

                progress.emit(ProgressEvent::Step {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        progress.stage_done(&denoise_label, denoise_start.elapsed());

        drop(cap_feats);
        drop(cap_mask);

        // Drop transformer to free VRAM for VAE decode
        loaded.transformer = None;
        loaded.device.synchronize()?;

        // 8. VAE decode — try GPU first, fall back to CPU on OOM
        progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        let latents_4d = latents.squeeze(2)?;

        let image = {
            let decode_latents = latents_4d.to_device(&loaded.vae_device)?.to_dtype(
                if loaded.vae_device.is_cpu() {
                    DType::F32
                } else {
                    loaded.dtype
                },
            )?;
            match loaded.vae.decode(&decode_latents) {
                Ok(img) => img,
                Err(e) if loaded.vae_device.is_cuda() => {
                    let err_msg = format!("{e}");
                    if err_msg.contains("OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                        tracing::warn!("VAE decode OOM on GPU, falling back to CPU");
                        progress.info("VAE decode OOM on GPU — retrying on CPU");
                        loaded.device.synchronize()?;
                        let vae_cfg = VaeConfig::z_image();
                        let vae_vb = unsafe {
                            VarBuilder::from_mmaped_safetensors(
                                &[loaded.vae_path.as_path()],
                                DType::F32,
                                &Device::Cpu,
                            )?
                        };
                        let cpu_vae = AutoEncoderKL::new(&vae_cfg, vae_vb)?;
                        let cpu_latents =
                            latents_4d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        cpu_vae.decode(&cpu_latents)?
                    } else {
                        return Err(e.into());
                    }
                }
                Err(e) => return Err(e.into()),
            }
        };

        let image = postprocess_image(&image)?;
        let image = image.i(0)?;

        progress.stage_done("VAE decode", vae_start.elapsed());

        let image_bytes = encode_image(&image, req.output_format, req.width, req.height)?;
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

        tracing::info!(model = %self.model_name, "loading Z-Image model components...");

        let is_gguf = self.detect_is_gguf();
        let text_tokenizer_path = self.validate_paths()?;

        let device =
            local_inference_helpers::device::create_device(|msg| self.progress.info(msg))?;
        let dtype = local_inference_helpers::dtype::gpu_dtype(&device);
        let transformer_cfg = Config::z_image_turbo();

        // Load transformer
        let xformer_label = if is_gguf {
            "Loading Z-Image transformer (GPU, quantized)".to_string()
        } else {
            let xformer_paths = self.transformer_paths();
            format!(
                "Loading Z-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg)?;
        self.progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        // Decide VAE placement based on remaining VRAM
        let free = free_vram_bytes().unwrap_or(0);
        if free > 0 {
            self.progress
                .info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
        }

        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let vae_dtype = if vae_on_gpu { dtype } else { DType::F32 };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        if !vae_on_gpu && (device.is_cuda() || device.is_metal()) {
            self.progress.info(&format!(
                "VAE on CPU ({} free < {} threshold for decode workspace)",
                fmt_gb(free),
                fmt_gb(VAE_DECODE_VRAM_THRESHOLD),
            ));
        }

        // Load VAE
        let vae_label = format!("Loading VAE ({vae_device_label})");
        self.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.progress.stage_done(&vae_label, vae_start.elapsed());

        // Qwen3 text encoder: auto-select variant based on VRAM
        self.progress.stage_start("Selecting Qwen3 encoder");
        let qwen3_resolve_start = Instant::now();
        let (resolved_paths, is_qwen3_gguf, te_on_gpu, te_device_label) = {
            let bf16_paths = self.paths.text_encoder_files.clone();
            let have_bf16 = !bf16_paths.is_empty() && bf16_paths.iter().all(|p| p.exists());
            encoders::variant_resolution::resolve_qwen3_variant(
                &self.progress,
                self.qwen3_variant.as_deref(),
                &device,
                free,
                &bf16_paths,
                have_bf16,
                false,
            )?
        };
        self.progress
            .stage_done("Selecting Qwen3 encoder", qwen3_resolve_start.elapsed());

        let te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_dtype = if te_on_gpu { dtype } else { DType::F32 };

        let te_label = if is_qwen3_gguf {
            format!("Loading Qwen3 text encoder (GGUF, {te_device_label})")
        } else {
            format!(
                "Loading Qwen3 text encoder ({} shards, {te_device_label})",
                resolved_paths.len(),
            )
        };
        self.progress.stage_start(&te_label);
        let te_start = Instant::now();

        let text_encoder = if is_qwen3_gguf {
            encoders::qwen3::Qwen3Encoder::load_gguf(
                &resolved_paths[0],
                &text_tokenizer_path,
                &te_device,
            )?
        } else {
            encoders::qwen3::Qwen3Encoder::load_bf16(
                &resolved_paths,
                &text_tokenizer_path,
                &te_device,
                te_dtype,
            )?
        };

        self.progress.stage_done(&te_label, te_start.elapsed());

        self.loaded = Some(LoadedZImage {
            transformer: Some(transformer),
            text_encoder,
            vae,
            transformer_cfg,
            device,
            vae_device,
            dtype,
            is_quantized: is_gguf,
            vae_path: self.paths.vae.clone(),
        });

        tracing::info!(model = %self.model_name, "all Z-Image components loaded successfully");
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn latent_dimensions() {
        assert_eq!(2 * (1024 / 16), 128);
        assert_eq!(2 * (512 / 16), 64);
        assert_eq!(2 * (768 / 16), 96);
    }
}
