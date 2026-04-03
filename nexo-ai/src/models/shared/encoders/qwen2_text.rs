//! Qwen2.5-VL text encoder for Qwen-Image-2512.
//!
//! Self-contained encoder implementing the Qwen2.5-VL text stack.
//! The Qwen-Image transformer expects conditioning in a specific layout:
//! - Chat-style prompt formatting with system prompt
//! - Padded text conditioning to a fixed 1024-token window
//! - Last hidden layer output (no final RMSNorm)
//!
//! Architecture: 28 layers, 28 Q heads, 4 KV heads (GQA 7:1),
//! hidden_size=3584, 18944 intermediate, RoPE theta=1e6, RMSNorm eps=1e-6.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::path::Path;
use std::sync::Arc;

const TEXT_WINDOW: usize = 1024;
const SYSTEM_PROMPT: &str = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

fn format_qwen_image_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn qwen_image_system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

// ── Config ──────────────────────────────────────────────────────────────────

struct Qwen2TextEncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
}

impl Qwen2TextEncoderConfig {
    fn qwen_image() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 128000,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

// ── Rotary Embedding ────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen2TextEncoderConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP (SwiGLU) ────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(cfg: &Qwen2TextEncoderConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&candle_nn::Activation::Silu)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── GQA Attention ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Qwen2TextEncoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        // Qwen2.5-VL attention uses biased Q/K/V projections, bias-free O projection
        let q_proj = candle_nn::linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
                .contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Qwen2TextEncoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.post_attention_layernorm)?
            .apply(&self.mlp)?;
        (residual + xs).map_err(Into::into)
    }
}

// ── Qwen2 Text Model ───────────────────────────────────────────────────────

struct Qwen2TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Qwen2TextModel {
    fn new(cfg: &Qwen2TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            sliding_window: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
            .map_err(Into::into)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = attn_mask.dims2()?;
        let mut mask = Vec::with_capacity(b_sz);
        for b in 0..b_sz {
            let token_mask = attn_mask.i((b, ..))?.expand((1, 1, seq_len, seq_len))?;
            mask.push(token_mask);
        }
        let mask = Tensor::cat(&mask.iter().collect::<Vec<_>>(), 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false).map_err(Into::into)
    }

    fn forward_last_hidden(
        &self,
        input_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, 0)?)
                }
            }
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        // Return the LAST hidden layer output (hidden_states[-1]),
        // matching diffusers pipeline_qwenimage.py.
        let target_layer = self.layers.len().saturating_sub(1);
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, attention_mask.as_ref(), 0)?;
            if idx == target_layer {
                return Ok(xs);
            }
        }
        anyhow::bail!("Qwen2 text model has too few layers")
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Loaded Qwen2.5-VL text encoder for Qwen-Image.
pub struct Qwen2TextEncoder {
    model: Qwen2TextModel,
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen2TextEncoder {
    /// Load from safetensors files.
    pub fn load(
        encoder_paths: &[impl AsRef<Path>],
        tokenizer_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let config = Qwen2TextEncoderConfig::qwen_image();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .filter_map(|p| p.as_ref().to_str())
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = Qwen2TextModel::new(&config, vb)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2.5 tokenizer: {e}"))?;
        Ok(Self { model, tokenizer })
    }

    fn encode_ids(&self, prompt: &str) -> Result<(Vec<u32>, usize)> {
        let prefix = qwen_image_system_prefix();
        let formatted = format_qwen_image_prompt(prompt);
        let prefix_ids = self
            .tokenizer
            .encode(prefix, false)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 prefix tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        let mut input_ids = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 tokenizer missing <|endoftext|>"))?;

        let drop_idx = prefix_ids.len();
        let full_window = TEXT_WINDOW + drop_idx;
        if input_ids.len() > full_window {
            input_ids.truncate(full_window);
        }
        let valid_len = input_ids.len().saturating_sub(drop_idx).min(TEXT_WINDOW);
        input_ids.resize(full_window, pad_id);
        Ok((input_ids, valid_len))
    }

    /// Encode a prompt, returning fixed-width embeddings and a matching mask
    /// after removing the system-prefix tokens. The resulting sequence length
    /// is always 1024.
    ///
    /// Returns `(hidden_states, attention_mask, valid_token_count)`.
    pub fn encode(
        &self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let (tokens, valid_len) = self.encode_ids(prompt)?;
        let drop_idx = tokens.len() - TEXT_WINDOW;

        let input_ids =
            Tensor::from_vec(tokens, (1, TEXT_WINDOW + drop_idx), &self.model.device)?;
        let mut mask = vec![0u8; TEXT_WINDOW + drop_idx];
        for value in &mut mask[..drop_idx + valid_len] {
            *value = 1;
        }
        let attn_mask =
            Tensor::from_vec(mask, (1, TEXT_WINDOW + drop_idx), &self.model.device)?;

        let emb = self
            .model
            .forward_last_hidden(&input_ids, Some(&attn_mask))?;
        let emb = emb.narrow(1, drop_idx, TEXT_WINDOW)?;

        let mut text_mask = vec![0u8; TEXT_WINDOW];
        for value in &mut text_mask[..valid_len] {
            *value = 1;
        }
        let text_mask = Tensor::from_vec(text_mask, (1, TEXT_WINDOW), &self.model.device)?;

        Ok((
            emb.to_device(target_device)?.to_dtype(target_dtype)?,
            text_mask.to_device(target_device)?,
            valid_len,
        ))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn prompt_formatting() {
        let result = format_qwen_image_prompt("a cat");
        assert!(result.starts_with("<|im_start|>system\n"));
        assert!(result.contains("a cat"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn system_prefix_does_not_include_user() {
        let prefix = qwen_image_system_prefix();
        assert!(prefix.contains(SYSTEM_PROMPT));
        assert!(!prefix.contains("user"));
    }

    #[test]
    fn config_dimensions() {
        let cfg = Qwen2TextEncoderConfig::qwen_image();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(head_dim, 128);
    }

    #[test]
    fn text_window_constant() {
        assert_eq!(TEXT_WINDOW, 1024);
    }
}
