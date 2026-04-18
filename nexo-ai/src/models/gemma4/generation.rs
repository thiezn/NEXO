//! Shared generation logic for Gemma 4 models (safetensors and GGUF).
//!
//! Contains the format-independent tokenization, sampling, and generation loop.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

// ── Stop tokens ──────────────────────────────────────────────────────────

/// Gemma 4 stop token names to look up in the tokenizer.
/// Reference: <https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4>
const EXTRA_STOP_TOKENS: &[&str] = &["<turn|>", "<|tool_response>"];

/// Build the stop-token-id list for Gemma 4 generation.
///
/// Starts with `config_eos` (from model config or GGUF metadata), then adds
/// `<turn|>` and `<|tool_response>` if the tokenizer knows them.
pub fn build_stop_token_ids(config_eos: u32, tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let mut ids = vec![config_eos];
    for &token_str in EXTRA_STOP_TOKENS {
        if let Some(id) = tokenizer.token_to_id(token_str) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

// ── Text forward trait ────────────────────────────────────────────────────

/// Minimal trait for text-only forward pass, implemented by both the
/// safetensors `TextModel` and the GGUF `QuantizedTextModel`.
pub trait TextForward {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle_core::Result<Tensor>;
    fn clear_kv_cache(&mut self);
}

// ── Prefix computation ────────────────────────────────────────────────────

/// Compute how many leading tokens are identical between two sequences.
pub fn compute_prefix_len(old_tokens: &[u32], new_tokens: &[u32]) -> usize {
    old_tokens
        .iter()
        .zip(new_tokens)
        .take_while(|(a, b)| a == b)
        .count()
}

// ── Tokenize with session prefix ──────────────────────────────────────────

/// Tokenize a prompt and compute the reusable prefix length for the given session.
/// Returns (tokens, prefix_len).
pub fn tokenize_with_prefix(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    processed_tokens: &[u32],
    current_session_id: Option<&str>,
    session_id: &Option<String>,
) -> Result<(Vec<u32>, usize)> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let new_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let prefix_len = if session_id.is_some() && session_id.as_deref() == current_session_id {
        let pl = compute_prefix_len(processed_tokens, &new_tokens);
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
                current_session_id,
            );
        } else {
            tracing::debug!("KV cache: no session_id, clearing cache");
        }
        0
    };

    Ok((new_tokens, prefix_len))
}

// ── Generation loop ───────────────────────────────────────────────────────

/// Run the autoregressive generation loop on any model implementing `TextForward`.
///
/// Returns `(decoded_text, tokens_generated, inference_time_ms)`.
pub fn generate(
    model: &mut dyn TextForward,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    stop_token_ids: &[u32],
    max_context_tokens: usize,
    mut tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    prefix_len: usize,
) -> Result<(String, usize, u64)> {
    let start = Instant::now();

    tracing::trace!(
        "generate: {} input tokens, max_tokens={}, prefix_len={}, temp={}, top_p={}, top_k={:?}",
        tokens.len(),
        max_tokens,
        prefix_len,
        temperature,
        top_p,
        top_k,
    );

    // Truncate to max context
    if tokens.len() > max_context_tokens {
        let excess = tokens.len() - max_context_tokens;
        tokens.drain(..excess);
        tracing::trace!("truncated {} tokens to fit context window", excess);
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
        Sampling::TopP {
            p: top_p,
            temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

    // Only clear KV cache when there's no prefix to reuse
    if prefix_len == 0 {
        tracing::trace!("clearing KV cache (no prefix reuse)");
        model.clear_kv_cache();
    }

    let mut generated_tokens = 0usize;
    let mut output_tokens: Vec<u32> = Vec::new();

    for index in 0..max_tokens {
        let context_size = if index > 0 {
            1
        } else {
            tokens.len() - prefix_len
        };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;

        if index == 0 {
            tracing::trace!(
                "prefill: {} tokens at start_pos={} (prefix_len={})",
                ctxt.len(),
                start_pos,
                prefix_len,
            );
        }

        let fwd_start = Instant::now();
        let logits = model.forward(&input, start_pos)?;
        let fwd_ms = fwd_start.elapsed().as_millis();

        if index == 0 {
            tracing::trace!("prefill forward pass took {}ms", fwd_ms);
        }

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

        if index < 3 {
            tracing::trace!("token[{}] = {} (fwd={}ms)", index, next_token, fwd_ms);
        }

        if stop_token_ids.contains(&next_token) {
            tracing::trace!("EOS at token {}", index);
            break;
        }

        output_tokens.push(next_token);
    }

    let text = tokenizer
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
