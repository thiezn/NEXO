use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use serde::Deserialize;

use crate::shared::types::*;

use super::gemma3_model;
use super::template;

// ── Config deserialization ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Gemma3HfTextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_query_pre_attn_scalar")]
    query_pre_attn_scalar: usize,
}

fn default_num_attention_heads() -> usize { 8 }
fn default_num_key_value_heads() -> usize { 4 }
fn default_head_dim() -> usize { 256 }
fn default_sliding_window() -> usize { 1024 }
fn default_query_pre_attn_scalar() -> usize { 256 }

#[derive(Debug, Deserialize)]
struct Gemma3HfConfig {
    text_config: Gemma3HfTextConfig,
    #[serde(default)]
    eos_token_id: Vec<u32>,
    vision_config: Option<serde_json::Value>,
}

impl Gemma3HfConfig {
    fn to_candle_config(&self) -> gemma3_model::Config {
        let tc = &self.text_config;
        gemma3_model::Config {
            attention_bias: false,
            head_dim: tc.head_dim,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            hidden_size: tc.hidden_size,
            intermediate_size: tc.intermediate_size,
            num_attention_heads: tc.num_attention_heads,
            num_hidden_layers: tc.num_hidden_layers,
            num_key_value_heads: tc.num_key_value_heads,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            vocab_size: 262_208,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: tc.query_pre_attn_scalar,
            sliding_window: tc.sliding_window,
            sliding_window_pattern: 6,
            max_position_embeddings: 131_072,
        }
    }
}

// ── Loaded state ────────────────────────────────────────────────────────────

pub struct LoadedState {
    pub model: gemma3_model::Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub eos_token_ids: Vec<u32>,
    pub vision: Option<super::vision::VisionState>,
}

// ── Load ────────────────────────────────────────────────────────────────────

pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let device = crate::device::create_device(|msg| tracing::info!("{msg}"))?;

    let config_path = model_dir.join("config.json");
    let config_str =
        std::fs::read_to_string(&config_path).context("failed to read config.json")?;
    let hf_config: Gemma3HfConfig =
        serde_json::from_str(&config_str).context("failed to parse config.json")?;
    let config = hf_config.to_candle_config();

    let safetensor_files = crate::models::shared::weights::find_safetensor_files(model_dir)?;
    tracing::info!(
        "loading {} safetensor file(s) from {}",
        safetensor_files.len(),
        model_dir.display()
    );

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, DType::BF16, &device)?
    };

    // Weights are under `language_model.model.*` — Model::new() internally adds `.model`.
    let vb_text = vb.pp("language_model");
    let model = gemma3_model::Model::new(&config, vb_text)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // Load vision components (SigLIP + projector) if vision_config is present.
    let vision = if hf_config.vision_config.is_some() {
        match super::vision::load_vision(&vb, &config_str, &device, &tokenizer) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("failed to load vision components: {e}");
                None
            }
        }
    } else {
        None
    };

    let eos_token_ids = if hf_config.eos_token_id.is_empty() {
        vec![1, 106] // fallback defaults
    } else {
        hf_config.eos_token_id
    };

    tracing::info!(
        "gemma3 model loaded (eos tokens: {:?}, vision: {})",
        eos_token_ids,
        vision.is_some()
    );
    Ok(LoadedState {
        model,
        tokenizer,
        device,
        eos_token_ids,
        vision,
    })
}

// ── Token sampling ──────────────────────────────────────────────────────────

pub fn create_sampler(temperature: f64, top_p: f64, seed: u64) -> LogitsProcessor {
    let temp = if temperature <= 1e-7 {
        None
    } else {
        Some(temperature)
    };
    LogitsProcessor::new(seed, temp, Some(top_p))
}

// ── Generation ──────────────────────────────────────────────────────────────

fn generate(
    state: &mut LoadedState,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> Result<(Vec<u32>, u64)> {
    let start = std::time::Instant::now();
    state.model.clear_kv_cache();
    let mut sampler = create_sampler(temperature, top_p, 0);

    let prompt_len = prompt_tokens.len();
    let input = Tensor::new(prompt_tokens, &state.device)?.unsqueeze(0)?;

    let logits = state.model.forward(&input, 0)?;
    let logits = logits.i((0, 0, ..))?;
    let mut next_token = sampler.sample(&logits)?;

    let mut generated = vec![next_token];

    if state.eos_token_ids.contains(&next_token) {
        let elapsed = start.elapsed().as_millis() as u64;
        return Ok((generated, elapsed));
    }

    for i in 0..max_tokens.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &state.device)?.unsqueeze(0)?;
        let seq_offset = prompt_len + i + 1;
        let logits = state.model.forward(&input, seq_offset)?;
        let logits = logits.i((0, 0, ..))?;
        next_token = sampler.sample(&logits)?;

        if state.eos_token_ids.contains(&next_token) {
            break;
        }
        generated.push(next_token);
    }

    let elapsed = start.elapsed().as_millis() as u64;
    Ok((generated, elapsed))
}

// ── Chat ────────────────────────────────────────────────────────────────────

pub fn chat(state: &mut LoadedState, request: &ChatRequest) -> Result<ChatResponse> {
    let prompt = template::format_chat_prompt(&request.messages);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let (generated, inference_time_ms) = generate(
        state,
        &prompt_tokens,
        request.max_tokens,
        request.temperature,
        request.top_p,
    )?;

    let text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

    Ok(ChatResponse {
        text,
        tokens_generated: generated.len(),
        inference_time_ms,
    })
}

// ── Tool calling ────────────────────────────────────────────────────────────

pub fn call_tools(state: &mut LoadedState, request: &ToolCallRequest) -> Result<ToolCallResponse> {
    let prompt = template::format_tool_prompt(&request.messages, &request.tools);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Use lower temperature for structured output.
    let temperature = if request.temperature > 0.0 {
        request.temperature.min(0.3)
    } else {
        0.1
    };

    let (generated, inference_time_ms) = generate(
        state,
        &prompt_tokens,
        request.max_tokens,
        temperature,
        0.95,
    )?;

    let text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

    let (tool_calls, reasoning) = template::parse_tool_response(&text);

    Ok(ToolCallResponse {
        tool_calls,
        reasoning,
        tokens_generated: generated.len(),
        inference_time_ms,
    })
}
