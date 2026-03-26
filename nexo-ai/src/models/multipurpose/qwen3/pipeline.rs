use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::shared::templates::kv_cache::KvCacheState;
use crate::shared::templates::{ChatTemplate, ReasoningMode};
use crate::shared::types::*;

use super::qwen3_dense;
use super::qwen3_moe;
use super::template::{Qwen3Template, strip_thinking, parse_tool_response};

// ── Model weights abstraction ────────────────────────────────────────────────

pub(super) enum Qwen3Weights {
    Dense(qwen3_dense::ModelWeights),
    Moe(qwen3_moe::ModelWeights),
}

impl Qwen3Weights {
    pub(super) fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(input, offset),
            Self::Moe(m) => m.forward(input, offset),
        }
    }

    pub(super) fn clear_kv_cache(&mut self) {
        match self {
            Self::Dense(m) => m.clear_kv_cache(),
            Self::Moe(m) => m.clear_kv_cache(),
        }
    }

    pub(super) fn embed_tokens(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.embed_tokens(input),
            Self::Moe(m) => m.embed_tokens(input),
        }
    }

    pub(super) fn forward_embeds(
        &mut self,
        xs: &Tensor,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward_embeds(xs, offset),
            Self::Moe(m) => m.forward_embeds(xs, offset),
        }
    }
}

impl KvCacheState for Qwen3Weights {
    fn cache_token_count(&self) -> usize {
        match self {
            Self::Dense(m) => m.cache_token_count(),
            Self::Moe(m) => m.cache_token_count(),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            Self::Dense(m) => m.clear_cache(),
            Self::Moe(m) => m.clear_cache(),
        }
    }

    fn truncate_to(&mut self, len: usize) {
        match self {
            Self::Dense(m) => m.truncate_to(len),
            Self::Moe(m) => m.truncate_to(len),
        }
    }
}

// ── Loaded state ─────────────────────────────────────────────────────────────

pub struct LoadedState {
    pub(super) weights: Qwen3Weights,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub eos_token_ids: Vec<u32>,
    cached_prompt_tokens: Vec<u32>,
    pub vision: Option<super::vision::VisionState>,
}

// ── Load ─────────────────────────────────────────────────────────────────────

pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let device = crate::device::create_device(|msg| tracing::info!("{msg}"))?;

    let gguf_path = crate::models::shared::weights::find_gguf_file(model_dir, "", &["mmproj"])?;
    tracing::info!("loading GGUF model from {}", gguf_path.display());

    let (content, mut file) = crate::models::shared::weights::load_gguf(&gguf_path)?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "qwen3".to_string());

    let is_moe = content
        .metadata
        .get(&format!("{arch}.expert_count"))
        .and_then(|v| v.to_u32().ok())
        .is_some_and(|n| n > 0);

    let weights = if is_moe {
        tracing::info!("detected MoE architecture ({arch})");
        let model =
            qwen3_moe::ModelWeights::from_gguf(content, &mut file, &device, DType::F32)?;
        Qwen3Weights::Moe(model)
    } else {
        tracing::info!("detected dense architecture ({arch})");
        let model = qwen3_dense::ModelWeights::from_gguf(content, &mut file, &device)?;
        Qwen3Weights::Dense(model)
    };

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let eos_token_ids = resolve_eos_tokens(&tokenizer);
    let vision = load_vision_if_available(model_dir, &device)?;

    tracing::info!(
        "qwen3 model loaded (eos tokens: {:?}, vision: {})",
        eos_token_ids,
        vision.is_some()
    );

    Ok(LoadedState {
        weights,
        tokenizer,
        device,
        eos_token_ids,
        cached_prompt_tokens: Vec::new(),
        vision,
    })
}

fn resolve_eos_tokens(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let mut eos_ids = Vec::new();
    for token_str in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(token_str) {
            eos_ids.push(id);
        }
    }
    if eos_ids.is_empty() {
        eos_ids.push(151643); // Qwen3 default <|im_end|>
    }
    eos_ids
}

fn load_vision_if_available(
    model_dir: &Path,
    device: &Device,
) -> Result<Option<super::vision::VisionState>> {
    let mmproj = crate::models::shared::weights::find_gguf_file(model_dir, "mmproj", &[]);
    match mmproj {
        Ok(path) => {
            tracing::info!("loading vision projector from {}", path.display());
            match super::vision::load_vision(&path, device) {
                Ok(v) => Ok(Some(v)),
                Err(e) => {
                    tracing::warn!("failed to load vision projector: {e}");
                    Ok(None)
                }
            }
        }
        Err(_) => Ok(None),
    }
}

// ── Token sampling ───────────────────────────────────────────────────────────

pub fn create_sampler(temperature: f64, top_p: f64, seed: u64) -> LogitsProcessor {
    let temp = if temperature <= 1e-7 {
        None
    } else {
        Some(temperature)
    };
    LogitsProcessor::new(seed, temp, Some(top_p))
}

// ── Generation ───────────────────────────────────────────────────────────────

fn generate(
    state: &mut LoadedState,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    max_context_tokens: Option<usize>,
) -> Result<(Vec<u32>, u64)> {
    if let Some(budget) = max_context_tokens {
        let total = prompt_tokens.len() + max_tokens;
        if total > budget {
            anyhow::bail!(
                "context budget exceeded: prompt ({}) + max_tokens ({}) = {} > budget ({}). \
                 Slide the conversation window to reduce prompt size.",
                prompt_tokens.len(),
                max_tokens,
                total,
                budget,
            );
        }
    }

    let start = std::time::Instant::now();
    let mut sampler = create_sampler(temperature, top_p, 0);

    // Compute common prefix between cached and new prompt tokens.
    let cached_len = state.weights.cache_token_count();
    let common_prefix = state
        .cached_prompt_tokens
        .iter()
        .zip(prompt_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let prefill_offset;
    let prefill_tokens: &[u32];

    if common_prefix > 0 && common_prefix == cached_len && cached_len == prompt_tokens.len() {
        // Exact match — truncate last token so we can re-forward it for logits.
        tracing::debug!(cached_len, "KV cache exact match");
        state.weights.truncate_to(cached_len - 1);
        prefill_tokens = &prompt_tokens[cached_len - 1..];
        prefill_offset = cached_len - 1;
    } else if common_prefix > 0 && common_prefix == cached_len {
        // Cache is a valid prefix of the new prompt — process only new tokens.
        tracing::debug!(
            common_prefix,
            new_tokens = prompt_tokens.len() - common_prefix,
            "KV cache prefix reuse"
        );
        prefill_tokens = &prompt_tokens[common_prefix..];
        prefill_offset = common_prefix;
    } else if common_prefix > 0 {
        // Partial match — truncate cache to common prefix, process remainder.
        tracing::debug!(
            common_prefix,
            cached_len,
            "KV cache partial match, truncating"
        );
        state.weights.truncate_to(common_prefix);
        prefill_tokens = &prompt_tokens[common_prefix..];
        prefill_offset = common_prefix;
    } else {
        // No match — full reset and full prefill.
        state.weights.clear_kv_cache();
        prefill_tokens = prompt_tokens;
        prefill_offset = 0;
    }

    let prompt_len = prompt_tokens.len();
    let input = Tensor::new(prefill_tokens, &state.device)?.unsqueeze(0)?;
    let logits = state.weights.forward(&input, prefill_offset)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = sampler.sample(&logits)?;

    let mut generated = vec![next_token];

    if state.eos_token_ids.contains(&next_token) {
        if state.cached_prompt_tokens != prompt_tokens {
            state.cached_prompt_tokens = prompt_tokens.to_vec();
        }
        let elapsed = start.elapsed().as_millis() as u64;
        return Ok((generated, elapsed));
    }

    for i in 0..max_tokens.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &state.device)?.unsqueeze(0)?;
        let seq_offset = prompt_len + i + 1;
        let logits = state.weights.forward(&input, seq_offset)?;
        let logits = logits.squeeze(0)?;
        next_token = sampler.sample(&logits)?;

        if state.eos_token_ids.contains(&next_token) {
            break;
        }
        generated.push(next_token);
    }

    if state.cached_prompt_tokens != prompt_tokens {
        state.cached_prompt_tokens = prompt_tokens.to_vec();
    }
    let elapsed = start.elapsed().as_millis() as u64;
    Ok((generated, elapsed))
}

// ── Chat ─────────────────────────────────────────────────────────────────────

pub fn chat(
    state: &mut LoadedState,
    request: &ChatRequest,
    max_context_tokens: Option<usize>,
) -> Result<ChatResponse> {
    let prompt = Qwen3Template.format_prompt(&request.messages, &ReasoningMode::Auto);
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
        max_context_tokens,
    )?;

    let raw_text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

    tracing::debug!(
        tokens = generated.len(),
        raw_text = %raw_text,
        "qwen3 chat raw output"
    );

    let text = strip_thinking(&raw_text);

    Ok(ChatResponse {
        text,
        tokens_generated: generated.len(),
        inference_time_ms,
    })
}

// ── Tool calling ─────────────────────────────────────────────────────────────

pub fn call_tools(
    state: &mut LoadedState,
    request: &ToolCallRequest,
    max_context_tokens: Option<usize>,
) -> Result<ToolCallResponse> {
    let prompt = Qwen3Template.format_with_tools(&request.messages, &request.tools, &ReasoningMode::Auto);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

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
        max_context_tokens,
    )?;

    let raw_text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

    let (tool_calls, reasoning) = parse_tool_response(&raw_text);

    Ok(ToolCallResponse {
        tool_calls,
        reasoning,
        tokens_generated: generated.len(),
        inference_time_ms,
    })
}
