#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::time::Instant;

use anyhow::Context;
use mlx_rs::{Array, Dtype, transforms};
use tokenizers::Tokenizer;

use crate::config::ModelPaths;
use crate::mlx_helpers::sampling::sample_token;
use crate::mlx_helpers::weight_loader::load_safetensors_shards;
use crate::model_config::Config;

use super::TextRequest;
use super::TextResponse;
use super::text::Qwen35TextModel;

struct LoadedState {
    text: Qwen35TextModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: Config,
}

pub struct Qwen35Engine {
    name: String,
    paths: ModelPaths,
    state: Option<LoadedState>,
}

impl Qwen35Engine {
    pub fn new(name: String, paths: ModelPaths) -> Self {
        Self {
            name,
            paths,
            state: None,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.name
    }

    pub fn load(&mut self) -> anyhow::Result<()> {
        tracing::info!("loading config from {}", self.paths.config_json.display());
        let config_data = std::fs::read_to_string(&self.paths.config_json)?;
        let config: Config = serde_json::from_str(&config_data)
            .context("failed to parse config.json as Qwen3.5 config")?;

        tracing::info!("loading tokenizer from {}", self.paths.tokenizer.display());
        let tokenizer = Tokenizer::from_file(&self.paths.tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        tracing::info!("loading model weights");
        let safetensors_paths = self.paths.all_safetensors();
        let weights = load_safetensors_shards(&safetensors_paths)?;

        tracing::info!("constructing text model");
        let text = Qwen35TextModel::new(&config.text_config, &weights)?;
        tracing::info!("text model loaded ({} layers)", config.text_config.num_hidden_layers);

        self.state = Some(LoadedState {
            text,
            tokenizer,
            config,
        });
        Ok(())
    }

    pub fn generate_text(&mut self, req: &TextRequest) -> anyhow::Result<TextResponse> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded, call load() first"))?;

        state.text.reset_caches();

        let input_ids = build_text_input_ids(&state.tokenizer, &req.prompt)?;
        let seq_len = input_ids.len();
        tracing::info!(seq_len, "built text input sequence");

        let input_ids_array = Array::from_slice(
            &input_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            &[1, seq_len as i32],
        );
        let input_embeds = state.text.embed_tokens(&input_ids_array)?;

        let (text, tokens_generated) = generate_loop(
            state,
            input_embeds,
            seq_len,
            req.max_tokens,
            req.temperature,
            req.top_p,
        )?;

        Ok(TextResponse {
            text,
            tokens_generated,
        })
    }
}

fn prepare_causal_mask(tgt_len: usize, dtype: Dtype) -> anyhow::Result<Array> {
    let mut mask_data = vec![0f32; tgt_len * tgt_len];
    for i in 0..tgt_len {
        for j in 0..tgt_len {
            if j > i {
                mask_data[i * tgt_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Array::from_slice(&mask_data, &[1, 1, tgt_len as i32, tgt_len as i32]);
    Ok(mask.as_dtype(dtype)?)
}

fn generate_loop(
    state: &mut LoadedState,
    input_embeds: Array,
    seq_len: usize,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> anyhow::Result<(String, usize)> {
    let start = Instant::now();

    // Create causal mask for prefill (only needed when seq_len > 1)
    let mask = if seq_len > 1 {
        Some(prepare_causal_mask(seq_len, Dtype::BFloat16)?)
    } else {
        None
    };

    tracing::info!(seq_len, "prefill starting");
    let prefill_start = Instant::now();
    let logits = state
        .text
        .forward_embeds(input_embeds, mask.as_ref(), 0)?;
    transforms::eval(&[&logits])?;
    tracing::info!(
        elapsed_ms = prefill_start.elapsed().as_millis() as u64,
        "prefill done"
    );

    let logits = logits.squeeze(&[0])?;
    let mut next_token = sample_token(&logits, temperature, top_p)?;

    let eos_token_id = state.tokenizer.token_to_id("<|endoftext|>");
    let im_end_token = state.tokenizer.token_to_id("<|im_end|>");

    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut current_pos = seq_len;

    for _ in 0..max_tokens {
        if Some(next_token) == eos_token_id || Some(next_token) == im_end_token {
            break;
        }

        generated_tokens.push(next_token);

        let token_array = Array::from_slice(&[next_token], &[1, 1]);
        let token_embeds = state.text.embed_tokens(&token_array)?;

        let logits = state.text.forward_embeds(token_embeds, None, current_pos)?;
        transforms::eval(&[&logits])?;

        let logits = logits.squeeze(&[0])?;
        next_token = sample_token(&logits, temperature, top_p)?;
        current_pos += 1;
    }

    let text = state
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("failed to decode tokens: {e}"))?;

    let tokens_generated = generated_tokens.len();
    let elapsed = start.elapsed().as_millis() as u64;
    tracing::info!(
        tokens = tokens_generated,
        elapsed_ms = elapsed,
        tok_per_sec = if elapsed > 0 {
            (tokens_generated as f64 / elapsed as f64 * 1000.0) as u64
        } else {
            0
        },
        "generation complete"
    );

    Ok((text, tokens_generated))
}

fn build_text_input_ids(tokenizer: &Tokenizer, prompt: &str) -> anyhow::Result<Vec<i64>> {
    let im_start = token_to_id(tokenizer, "<|im_start|>")
        .ok_or_else(|| anyhow::anyhow!("missing <|im_start|> token"))?;
    let im_end = token_to_id(tokenizer, "<|im_end|>")
        .ok_or_else(|| anyhow::anyhow!("missing <|im_end|> token"))?;
    let nl = encode_text(tokenizer, "\n")?;

    let mut ids: Vec<i64> = Vec::new();

    // System message
    ids.push(im_start);
    ids.extend(encode_text(tokenizer, "system")?);
    ids.extend(&nl);
    ids.extend(encode_text(tokenizer, "You are a helpful assistant.")?);
    ids.push(im_end);
    ids.extend(&nl);

    // User message
    ids.push(im_start);
    ids.extend(encode_text(tokenizer, "user")?);
    ids.extend(&nl);
    ids.extend(encode_text(tokenizer, prompt)?);
    ids.push(im_end);
    ids.extend(&nl);

    // Assistant turn start
    ids.push(im_start);
    ids.extend(encode_text(tokenizer, "assistant")?);
    ids.extend(&nl);

    Ok(ids)
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
}

fn token_to_id(tokenizer: &Tokenizer, token: &str) -> Option<i64> {
    tokenizer.token_to_id(token).map(i64::from)
}
