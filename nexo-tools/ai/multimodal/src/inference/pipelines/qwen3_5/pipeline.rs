#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::time::Instant;

use anyhow::Context;
use local_inference_helpers::candle_core::{DType, Device, Tensor, D};
use local_inference_helpers::candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::config::ModelPaths;
use crate::inference::{DescribeRequest, DescribeResponse, InferenceEngine};

use super::config::Config;
use super::sampling::sample_token;
use super::text::Qwen35TextModel;
use super::vision::Qwen35VisionModel;

struct LoadedState {
    text: Qwen35TextModel,
    vision: Qwen35VisionModel,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
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

}

fn prepare_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    num_attn_heads: usize,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mask: Vec<_> = (0..tgt_len)
        .flat_map(|i| {
            (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 })
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    Ok(mask
        .expand((b_size, num_attn_heads, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)?)
}

impl InferenceEngine for Qwen35Engine {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_loaded(&self) -> bool {
        self.state.is_some()
    }

    fn load(&mut self, device: &Device, dtype: DType) -> anyhow::Result<()> {
        tracing::info!("loading config from {}", self.paths.config_json.display());
        let config_data = std::fs::read_to_string(&self.paths.config_json)?;
        let config: Config = serde_json::from_str(&config_data)
            .context("failed to parse config.json as Qwen3.5 config")?;

        tracing::info!("loading tokenizer from {}", self.paths.tokenizer.display());
        let tokenizer = Tokenizer::from_file(&self.paths.tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        tracing::info!("loading model weights");
        let safetensors_paths = self.paths.all_safetensors();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_paths, dtype, device)
                .context("failed to load safetensors")?
        };

        let vision =
            Qwen35VisionModel::new(&config.vision_config, vb.pp("model").pp("visual"))?;
        tracing::info!("vision encoder loaded");

        let text = Qwen35TextModel::new(&config.text_config, vb.clone())?;
        tracing::info!("text model loaded");

        self.state = Some(LoadedState {
            text,
            vision,
            tokenizer,
            config,
            device: device.clone(),
        });
        Ok(())
    }

    fn describe(&mut self, req: &DescribeRequest) -> anyhow::Result<DescribeResponse> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded, call load() first"))?;

        let start = Instant::now();

        // Reset caches for new generation
        state.text.reset_kv_caches();

        // Build input token sequence
        let input_ids = build_input_ids(
            &state.tokenizer,
            &state.config,
            &req.prompt,
            req.num_image_tokens,
        )?;

        let seq_len = input_ids.len();
        tracing::info!(
            seq_len,
            image_tokens = req.num_image_tokens,
            "built input sequence"
        );

        // Find image placeholder spans
        let image_token_id = state.config.image_token_id as i64;
        let img_pad_spans = find_continuous_spans(&input_ids, image_token_id);

        // Run vision encoder
        let (image_embeds, _deepstack) =
            state
                .vision
                .forward(&req.pixel_values, &req.image_grid_thw)?;
        let image_embeds = image_embeds
            .to_device(&state.device)?
            .to_dtype(state.text.dtype)?;

        // Create input embeddings and inject vision features
        let input_ids_tensor =
            Tensor::from_vec(input_ids, (1, seq_len), &state.device)?;
        let mut input_embeds = state.text.embed_tokens(&input_ids_tensor)?;
        let (_, _, hidden_dim) = input_embeds.dims3()?;

        let mut offset = 0usize;
        for &(span_start, span_end) in &img_pad_spans {
            let len = span_end - span_start;
            let chunk = image_embeds.narrow(0, offset, len)?;
            offset += len;
            input_embeds = input_embeds.slice_assign(
                &[0..1, span_start..span_end, 0..hidden_dim],
                &chunk.unsqueeze(0)?,
            )?;
        }

        // Prefill: forward pass with full sequence
        let num_attn_heads = state.text.num_attn_heads;
        let dtype = state.text.dtype;
        let device = state.device.clone();
        let attention_mask = if seq_len > 1 {
            None
        } else {
            Some(prepare_attention_mask(
                1,
                seq_len,
                0,
                num_attn_heads,
                dtype,
                &device,
            )?)
        };

        let logits = state.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            0, // seqlen_offset
        )?;

        let last_logits = logits.squeeze(0)?;
        let mut next_token = sample_token(&last_logits, req.temperature, req.top_p)?;

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(req.max_tokens);
        let eos_token_id = state.tokenizer.token_to_id("<|endoftext|>");
        let im_end_token = state.tokenizer.token_to_id("<|im_end|>");

        let mut current_pos = seq_len;

        for _ in 0..req.max_tokens {
            if Some(next_token) == eos_token_id || Some(next_token) == im_end_token {
                break;
            }

            generated_tokens.push(next_token);

            // Single-token forward pass
            let token_tensor = Tensor::from_vec(
                vec![next_token as i64],
                (1, 1),
                &state.device,
            )?;
            let token_embeds = state.text.embed_tokens(&token_tensor)?;

            let logits = state.text.forward_embeds(
                token_embeds,
                None, // no mask needed for single token with KV cache
                current_pos,
            )?;

            let logits = logits.squeeze(0)?;
            next_token = sample_token(&logits, req.temperature, req.top_p)?;
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
            "generation complete"
        );

        Ok(DescribeResponse {
            text,
            tokens_generated,
        })
    }
}

fn build_input_ids(
    tokenizer: &Tokenizer,
    config: &Config,
    prompt: &str,
    num_image_tokens: usize,
) -> anyhow::Result<Vec<i64>> {
    let im_start = token_to_id(tokenizer, "<|im_start|>")
        .ok_or_else(|| anyhow::anyhow!("missing <|im_start|> token"))?;
    let im_end = token_to_id(tokenizer, "<|im_end|>")
        .ok_or_else(|| anyhow::anyhow!("missing <|im_end|> token"))?;
    let nl = encode_text(tokenizer, "\n")?;

    let image_token_id = i64::from(config.image_token_id);
    let vision_start = i64::from(config.vision_start_token_id);
    let vision_end = i64::from(config.vision_end_token_id);

    let mut ids: Vec<i64> = Vec::new();

    // System message
    ids.push(im_start);
    ids.extend(encode_text(tokenizer, "system")?);
    ids.extend(&nl);
    ids.extend(encode_text(tokenizer, "You are a helpful assistant.")?);
    ids.push(im_end);
    ids.extend(&nl);

    // User message with image
    ids.push(im_start);
    ids.extend(encode_text(tokenizer, "user")?);
    ids.extend(&nl);
    ids.push(vision_start);
    ids.resize(ids.len() + num_image_tokens, image_token_id);
    ids.push(vision_end);
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

fn find_continuous_spans(input_ids: &[i64], token_id: i64) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < input_ids.len() {
        if input_ids[i] == token_id {
            let start = i;
            while i < input_ids.len() && input_ids[i] == token_id {
                i += 1;
            }
            spans.push((start, i));
        } else {
            i += 1;
        }
    }
    spans
}
