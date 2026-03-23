use std::time::Instant;

use anyhow::Result;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::parler_tts;
use local_inference_helpers::candle_core::{Device, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use local_inference_helpers::progress::{ProgressCallback, ProgressReporter};

use crate::config::TTSModelPaths;
use crate::inference::{InferenceEngine, LoadStrategy, TTSRequest, TTSResponse};

/// Normalize a Parler-TTS config.json to ensure its `audio_encoder` section
/// matches the schema expected by candle-transformers' DAC `Config` struct.
///
/// Parler-TTS mini v1.1 uses a newer DAC config format with `n_codebooks`,
/// `downsampling_ratios`, `hop_length`, etc. instead of the older format's
/// `num_codebooks`, `frame_rate`, `latent_dim`, `model_bitrate`.
fn normalize_config(config_str: &str) -> Result<String> {
    // Fast path: if all required DAC fields are present, skip the parse/serialize round-trip
    if config_str.contains("\"num_codebooks\"")
        && config_str.contains("\"frame_rate\"")
        && config_str.contains("\"latent_dim\"")
        && config_str.contains("\"model_bitrate\"")
    {
        return Ok(config_str.to_string());
    }

    let mut root: serde_json::Value = serde_json::from_str(config_str)?;

    if let Some(ae) = root.get_mut("audio_encoder").and_then(|v| v.as_object_mut()) {
        if ae.get("num_codebooks").is_none() {
            if let Some(val) = ae.remove("n_codebooks") {
                ae.insert("num_codebooks".to_string(), val);
            }
        }

        let frame_rate = if ae.get("frame_rate").is_none() {
            if let (Some(sr), Some(ratios)) = (
                ae.get("sampling_rate").and_then(|v| v.as_u64()),
                ae.get("downsampling_ratios").and_then(|v| v.as_array()),
            ) {
                let product: u64 = ratios.iter().filter_map(|r| r.as_u64()).product();
                if product > 0 {
                    let fr = sr / product;
                    ae.insert("frame_rate".to_string(), serde_json::json!(fr));
                    Some(fr)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            ae.get("frame_rate").and_then(|v| v.as_u64())
        };

        if ae.get("latent_dim").is_none() {
            if let Some(val) = ae.remove("hidden_size") {
                ae.insert("latent_dim".to_string(), val);
            }
        }

        if ae.get("model_bitrate").is_none() {
            if let (Some(n_cb), Some(cb_size), Some(fr)) = (
                ae.get("num_codebooks").and_then(|v| v.as_u64()),
                ae.get("codebook_size").and_then(|v| v.as_u64()),
                frame_rate,
            ) {
                let bits_per_second = n_cb * (cb_size as f64).log2() as u64 * fr;
                ae.insert("model_bitrate".to_string(), serde_json::json!(bits_per_second / 1000));
            }
        }
    }

    Ok(serde_json::to_string(&root)?)
}

struct LoadedParler {
    model: parler_tts::Model,
    tokenizer: tokenizers::Tokenizer,
    config: parler_tts::Config,
    device: Device,
}

pub struct ParlerEngine {
    loaded: Option<LoadedParler>,
    model_name: String,
    paths: TTSModelPaths,
    progress: ProgressReporter,
    #[allow(dead_code)]
    load_strategy: LoadStrategy,
}

impl ParlerEngine {
    pub fn new(
        model_name: String,
        paths: TTSModelPaths,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            progress: ProgressReporter::default(),
            load_strategy,
        }
    }
}

impl InferenceEngine for ParlerEngine {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        let start = Instant::now();
        self.progress.stage_start("device");
        let device = local_inference_helpers::device::create_device(|msg| {
            tracing::info!("{msg}");
        })?;
        let dtype = local_inference_helpers::dtype::gpu_dtype(&device);
        self.progress
            .stage_done("device", start.elapsed());

        self.progress.stage_start("config");
        let config_str = std::fs::read_to_string(&self.paths.config_json)?;
        let config: parler_tts::Config = serde_json::from_str(&normalize_config(&config_str)?)?;
        self.progress
            .stage_done("config", start.elapsed());

        self.progress.stage_start("model");
        let safetensor_files: Vec<_> = self
            .paths
            .safetensor_files()
            .into_iter()
            .map(|p| p.to_path_buf())
            .collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)?
        };
        let model = parler_tts::Model::new(&config, vb)?;
        self.progress
            .stage_done("model", start.elapsed());

        self.progress.stage_start("tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(&self.paths.tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        self.progress
            .stage_done("tokenizer", start.elapsed());

        self.loaded = Some(LoadedParler {
            model,
            tokenizer,
            config,
            device,
        });

        tracing::info!("Parler-TTS model loaded in {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }

    fn synthesize(&mut self, req: &TTSRequest) -> Result<TTSResponse> {
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded. Call load() first."))?;

        let start = Instant::now();

        self.progress.stage_start("tokenize");
        let prompt_encoding = loaded
            .tokenizer
            .encode(req.text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let prompt_tokens =
            Tensor::new(prompt_encoding.get_ids(), &loaded.device)?.unsqueeze(0)?;

        let desc_encoding = loaded
            .tokenizer
            .encode(req.description.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Description tokenization failed: {e}"))?;
        let description_tokens =
            Tensor::new(desc_encoding.get_ids(), &loaded.device)?.unsqueeze(0)?;
        self.progress
            .stage_done("tokenize", start.elapsed());

        self.progress.stage_start("generate");
        let temperature = if req.temperature <= 0.0 {
            None
        } else {
            Some(req.temperature)
        };
        let lp = LogitsProcessor::new(req.seed, temperature, None);
        let audio_tokens =
            loaded
                .model
                .generate(&prompt_tokens, &description_tokens, lp, req.max_tokens)?;
        self.progress
            .stage_done("generate", start.elapsed());

        self.progress.stage_start("decode_audio");
        let pcm_tensor = loaded.model.audio_encoder.decode_codes(&audio_tokens)?;
        self.progress
            .stage_done("decode_audio", start.elapsed());

        let pcm_tensor = pcm_tensor.squeeze(0)?.squeeze(0)?;
        let pcm_samples: Vec<f32> = pcm_tensor.to_vec1()?;
        let sample_rate = loaded.config.audio_encoder.sampling_rate;
        let generation_time_ms = start.elapsed().as_millis() as u64;

        tracing::info!(
            "Generated {:.1}s of audio in {:.1}s (sample_rate={})",
            pcm_samples.len() as f64 / sample_rate as f64,
            generation_time_ms as f64 / 1000.0,
            sample_rate
        );

        Ok(TTSResponse {
            pcm_samples,
            sample_rate,
            generation_time_ms,
            model: self.model_name.clone(),
            seed_used: req.seed,
        })
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
