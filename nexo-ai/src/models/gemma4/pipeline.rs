use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use super::model::{
    Model,
    config::{Gemma4Config, Gemma4TextConfig},
    text::{LayerKvSnapshot, TextModel},
};

use crate::models::shared::weights::find_safetensor_files;
use crate::shared::templates::{ChatTemplate, ReasoningMode};
use crate::shared::types::{
    ChatRequest, ChatResponse, ImageAnalysisRequest, ImageAnalysisResponse, ToolCallRequest,
    ToolCallResponse,
};

use super::template::Gemma4Template;

// ── Loaded state ───────────────────────────────────────────────────────────

enum ModelKind {
    TextOnly(TextModel),
    Multimodal(Model),
}

pub struct LoadedState {
    model: ModelKind,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
    eos_token_id: u32,
    max_context_tokens: usize,
    current_session_id: Option<String>,
    processed_tokens: Vec<u32>,
}

impl LoadedState {
    pub fn current_session_id(&self) -> Option<&str> {
        self.current_session_id.as_deref()
    }

    pub fn processed_tokens(&self) -> &[u32] {
        &self.processed_tokens
    }

    pub fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>) {
        self.current_session_id = session_id;
        self.processed_tokens = tokens;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn kv_cache_seq_len(&self) -> usize {
        match &self.model {
            ModelKind::TextOnly(m) => m.kv_cache_seq_len(),
            ModelKind::Multimodal(m) => m.kv_cache_seq_len(),
        }
    }

    pub fn save_kv_cache(&self) -> candle_core::Result<Vec<LayerKvSnapshot>> {
        match &self.model {
            ModelKind::TextOnly(m) => m.save_kv_cache(),
            ModelKind::Multimodal(m) => m.save_kv_cache(),
        }
    }

    pub fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> candle_core::Result<()> {
        match &mut self.model {
            ModelKind::TextOnly(m) => m.restore_kv_cache(snapshots),
            ModelKind::Multimodal(m) => m.restore_kv_cache(snapshots),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.model {
            ModelKind::TextOnly(m) => m.clear_kv_cache(),
            ModelKind::Multimodal(m) => m.clear_kv_cache(),
        }
    }
}

// ── Load ───────────────────────────────────────────────────────────────────

/// Load a Gemma 4 model from the given directory.
///
/// Starts as text-only for faster loading and lower memory. The multimodal
/// towers are loaded on-demand when `analyze_image` is first called.
pub fn load(model_dir: &Path, max_context_tokens: Option<usize>) -> Result<LoadedState> {
    let start = Instant::now();

    let device = crate::device::create_device(|msg| {
        tracing::info!("{msg}");
    })?;
    let dtype = crate::device::gpu_compute_dtype(&device);
    tracing::info!("device ready in {:.1}s", start.elapsed().as_secs_f64());

    // Parse config — extract text_config for text-only loading
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", config_path.display()))?;
    let raw: serde_json::Value = serde_json::from_str(&config_str)?;
    let text_config: Gemma4TextConfig = if let Some(text_cfg) = raw.get("text_config") {
        serde_json::from_value(text_cfg.clone())?
    } else {
        serde_json::from_value(raw)?
    };

    tracing::info!(
        "config: hidden_size={}, layers={}, heads={}, vocab={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.num_attention_heads,
        text_config.vocab_size,
    );

    // Load weights
    let safetensor_files = find_safetensor_files(model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)? };

    // Load text model from the language_model sub-path in weights
    let text_model = TextModel::new(&text_config, vb.pp("model").pp("language_model"))?;
    tracing::info!(
        "text model loaded in {:.1}s ({} safetensor file(s))",
        start.elapsed().as_secs_f64(),
        safetensor_files.len()
    );

    // Tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // EOS token: </s> (token id 1)
    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<eos>"))
        .unwrap_or(1);

    let max_ctx = max_context_tokens.unwrap_or(text_config.max_position_embeddings);

    tracing::info!(
        "Gemma 4 fully loaded in {:.1}s (eos={}, max_ctx={})",
        start.elapsed().as_secs_f64(),
        eos_token_id,
        max_ctx,
    );

    Ok(LoadedState {
        model: ModelKind::TextOnly(text_model),
        tokenizer,
        device,
        dtype,
        eos_token_id,
        max_context_tokens: max_ctx,
        current_session_id: None,
        processed_tokens: Vec::new(),
    })
}

// ── Generation core ────────────────────────────────────────────────────────

/// Compute how many leading tokens are identical between two sequences.
fn compute_prefix_len(old_tokens: &[u32], new_tokens: &[u32]) -> usize {
    old_tokens
        .iter()
        .zip(new_tokens)
        .take_while(|(a, b)| a == b)
        .count()
}

fn generate(
    state: &mut LoadedState,
    mut tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    prefix_len: usize,
) -> Result<(String, usize, u64)> {
    let start = Instant::now();

    // Truncate to max context
    if tokens.len() > state.max_context_tokens {
        let excess = tokens.len() - state.max_context_tokens;
        tokens.drain(..excess);
    }

    let sampling = if temperature <= 0.0 {
        Sampling::ArgMax
    } else if let Some(k) = top_k {
        Sampling::TopKThenTopP {
            k: k as usize,
            p: top_p,
            temperature,
        }
    } else {
        Sampling::TopP { p: top_p, temperature }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

    // Only clear KV cache when there's no prefix to reuse
    if prefix_len == 0 {
        match &mut state.model {
            ModelKind::TextOnly(m) => m.clear_kv_cache(),
            ModelKind::Multimodal(m) => m.clear_kv_cache(),
        }
    }

    let mut generated_tokens = 0usize;
    let mut output_tokens: Vec<u32> = Vec::new();

    for index in 0..max_tokens {
        let context_size = if index > 0 { 1 } else { tokens.len() - prefix_len };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &state.device)?.unsqueeze(0)?;

        let logits = match &mut state.model {
            ModelKind::TextOnly(m) => m.forward(&input, start_pos)?,
            ModelKind::Multimodal(m) => m.forward(&input, start_pos)?,
        };

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        // Apply repeat penalty
        let logits = if !tokens.is_empty() {
            let penalty_start = tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                1.1,
                &tokens[penalty_start..],
            )?
        } else {
            logits
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if next_token == state.eos_token_id {
            break;
        }

        output_tokens.push(next_token);
    }

    let text = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))?;

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        "generated {} tokens in {}ms ({:.1} tok/s)",
        generated_tokens,
        inference_time_ms,
        generated_tokens as f64 / (inference_time_ms as f64 / 1000.0),
    );

    Ok((text, generated_tokens, inference_time_ms))
}

/// Tokenize a prompt and compute the reusable prefix length for the given session.
/// Returns (tokens, prefix_len). Also updates session tracking after the caller
/// finishes generation.
fn tokenize_with_prefix(
    state: &LoadedState,
    prompt: &str,
    session_id: &Option<String>,
) -> Result<(Vec<u32>, usize)> {
    let encoding = state
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let new_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let prefix_len = if session_id.is_some() && *session_id == state.current_session_id {
        let pl = compute_prefix_len(&state.processed_tokens, &new_tokens);
        if pl > 0 {
            tracing::debug!(
                "KV cache hit: reusing {pl}/{} tokens ({:.0}% prefill saved)",
                new_tokens.len(),
                pl as f64 / new_tokens.len() as f64 * 100.0,
            );
        } else {
            tracing::debug!("KV cache miss: same session but tokens diverged, clearing cache");
        }
        pl
    } else {
        if session_id.is_some() {
            tracing::debug!(
                "KV cache: new session {:?} (previous: {:?}), no cache to reuse",
                session_id,
                state.current_session_id,
            );
        } else {
            tracing::debug!("KV cache: no session_id, clearing cache");
        }
        0
    };

    Ok((new_tokens, prefix_len))
}

// ── Chat ───────────────────────────────────────────────────────────────────

pub fn chat(state: &mut LoadedState, request: &ChatRequest) -> Result<ChatResponse> {
    let template = Gemma4Template;
    let prompt = template.format_prompt(&request.messages, &ReasoningMode::Disabled);

    let (new_tokens, prefix_len) = tokenize_with_prefix(state, &prompt, &request.session_id)?;

    let (raw, tokens_generated, inference_time_ms) =
        generate(state, new_tokens.clone(), request.max_tokens, request.temperature, request.top_p, request.top_k, prefix_len)?;

    state.current_session_id = request.session_id.clone();
    state.processed_tokens = new_tokens;

    let text = template.parse_response(&raw);

    Ok(ChatResponse {
        text,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Tool calling ───────────────────────────────────────────────────────────

pub fn call_tools(state: &mut LoadedState, request: &ToolCallRequest) -> Result<ToolCallResponse> {
    let template = Gemma4Template;
    let prompt =
        template.format_with_tools(&request.messages, &request.tools, &ReasoningMode::Disabled);

    let (new_tokens, prefix_len) = tokenize_with_prefix(state, &prompt, &request.session_id)?;

    let (raw, tokens_generated, inference_time_ms) =
        generate(state, new_tokens.clone(), request.max_tokens, request.temperature, request.top_p, request.top_k, prefix_len)?;

    state.current_session_id = request.session_id.clone();
    state.processed_tokens = new_tokens;

    let (tool_calls, reasoning) = template.parse_tool_calls(&raw);

    Ok(ToolCallResponse {
        tool_calls,
        reasoning,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Image analysis ─────────────────────────────────────────────────────────

/// Analyze an image. On first call, upgrades from text-only to multimodal.
pub fn analyze_image(
    state: &mut LoadedState,
    request: &ImageAnalysisRequest,
    model_dir: &Path,
) -> Result<ImageAnalysisResponse> {
    // Upgrade to multimodal if currently text-only
    ensure_multimodal(state, model_dir)?;

    let start = Instant::now();

    // Decode image to pixel tensor
    let image = image::load_from_memory(&request.image_data)
        .map_err(|e| anyhow::anyhow!("failed to decode image: {e}"))?;
    let pixel_values = preprocess_image(&image, &state.device, state.dtype)?;

    // Build prompt with image token
    let prompt = format!(
        "<start_of_turn>user\n<start_of_image><end_of_image>{}<end_of_turn>\n<start_of_turn>model\n",
        request.prompt
    );

    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let mut tokens = encoding.get_ids().to_vec();

    let sampling = if request.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: 0.95,
            temperature: request.temperature,
        }
    };
    // Image analysis uses fixed sampling (no top_k override)
    let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

    // Clear KV cache
    if let ModelKind::Multimodal(m) = &mut state.model {
        m.clear_kv_cache();
    }

    let mut generated_tokens = 0usize;
    let mut output_tokens: Vec<u32> = Vec::new();

    for index in 0..request.max_tokens {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &state.device)?.unsqueeze(0)?;

        let logits = if let ModelKind::Multimodal(m) = &mut state.model {
            if index == 0 {
                // First pass: include image
                m.forward_multimodal(
                    &input,
                    Some(std::slice::from_ref(&pixel_values)),
                    None,
                    None,
                    start_pos,
                )?
            } else {
                m.forward(&input, start_pos)?
            }
        } else {
            anyhow::bail!("model should be multimodal for image analysis");
        };

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let logits = if !tokens.is_empty() {
            let penalty_start = tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                1.1,
                &tokens[penalty_start..],
            )?
        } else {
            logits
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if next_token == state.eos_token_id {
            break;
        }

        output_tokens.push(next_token);
    }

    let raw = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))?;

    let template = Gemma4Template;
    let text = template.parse_response(&raw);

    let inference_time_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        "image analysis: {} tokens in {}ms",
        generated_tokens,
        inference_time_ms,
    );

    Ok(ImageAnalysisResponse {
        text,
        tokens_generated: generated_tokens,
        inference_time_ms,
    })
}

/// Upgrade from text-only to multimodal by reloading weights with full config.
fn ensure_multimodal(state: &mut LoadedState, model_dir: &Path) -> Result<()> {
    if matches!(state.model, ModelKind::Multimodal(_)) {
        return Ok(());
    }

    tracing::info!("upgrading to multimodal model for image analysis...");
    let start = Instant::now();

    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Gemma4Config = serde_json::from_str(&config_str)?;

    let safetensor_files = find_safetensor_files(model_dir)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, state.dtype, &state.device)?
    };

    let model = Model::new(&config, vb)?;
    state.model = ModelKind::Multimodal(model);

    tracing::info!(
        "multimodal upgrade complete in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Preprocess an image for Gemma 4 vision tower.
///
/// Resizes to 896x896, normalizes with ImageNet stats, returns `(1, 3, H, W)`.
fn preprocess_image(image: &image::DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let target_size = 896u32;

    let resized = image.resize_exact(
        target_size,
        target_size,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw_pixels = rgb.into_raw();

    // Convert HWC u8 -> CHW f32, normalize with ImageNet stats
    let mean = [0.5f32, 0.5, 0.5];
    let std = [0.5f32, 0.5, 0.5];

    let mut chw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            for c in 0..3 {
                let pixel = raw_pixels[src + c] as f32 / 255.0;
                chw[c * h * w + y * w + x] = (pixel - mean[c]) / std[c];
            }
        }
    }

    let tensor = Tensor::from_vec(chw, (1, 3, h, w), device)?.to_dtype(dtype)?;
    Ok(tensor)
}
