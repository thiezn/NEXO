//! Qwen3 text encoder for Flux.2 image generation.
//!
//! Custom implementation that extracts hidden states from specific intermediate
//! layers (9, 18, 27) and concatenates them to produce the 7680-dim embeddings
//! expected by the Flux.2 transformer (`context_in_dim = 2560 * 3`).
//!
//! Architecture matches the Qwen3 text encoder shipped with Flux.2 Klein/Dev:
//! `hidden_size=2560`, 36 layers, GQA (32 heads / 8 KV heads), SwiGLU MLP,
//! per-head RMSNorm on Q/K, RoPE with `theta=1_000_000`.

use anyhow::Result;
use candle_core::{DType, Device, Module, Result as CResult, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::utils::repeat_kv;
use std::path::Path;
use std::sync::Arc;

// ── Config ───────────────────────────────────────────────────────────────────

struct Qwen3Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_position_embeddings: usize,
}

impl Qwen3Config {
    fn flux() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
        }
    }
}

// ── Rotary Embedding ─────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> CResult<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CResult<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP (SwiGLU) ─────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gate = x.apply(&self.gate_proj)?.apply(&Activation::Silu)?;
        let up = x.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ── GQA Attention ────────────────────────────────────────────────────────────

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(cfg: &Qwen3Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> CResult<Self> {
        let h = cfg.num_attention_heads;
        let kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(cfg.hidden_size, h * hd, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(cfg.hidden_size, kv * hd, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(cfg.hidden_size, kv * hd, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(h * hd, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: h,
            num_kv_heads: kv,
            head_dim: hd,
            rotary,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let n_kv_groups = self.num_heads / self.num_kv_heads;

        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary.apply(&q, &k)?;

        let k = repeat_kv(k, n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, n_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let hidden_size = self.num_heads * self.head_dim;
        ctx.transpose(1, 2)?
            .reshape((b, l, hidden_size))?
            .apply(&self.o_proj)
    }
}

// ── Decoder Layer ────────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Qwen3Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CResult<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        x + h
    }
}

// ── Qwen3 Text Encoder ──────────────────────────────────────────────────────

struct Qwen3TextEncoder {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    device: Device,
    dtype: DType,
}

impl Qwen3TextEncoder {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> CResult<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let vb_layers = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn causal_mask(&self, b: usize, l: usize) -> CResult<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, l, l), &self.device)?.to_dtype(self.dtype)
    }

    /// Run the encoder and capture hidden states at specified layer indices.
    ///
    /// Returns the concatenation of hidden states from `layer_indices` along
    /// the last dimension. E.g. for layers [9, 18, 27] with hidden_size=2560,
    /// output shape is `(B, seq_len, 7680)`.
    ///
    /// Exits early after the last requested layer to save compute.
    fn forward_with_layers(&self, input_ids: &Tensor, layer_indices: &[usize]) -> CResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        let mask = if l == 1 { None } else { Some(self.causal_mask(b, l)?) };

        let max_layer = layer_indices.iter().copied().max().unwrap_or(0);
        let mut captured: Vec<Tensor> = Vec::with_capacity(layer_indices.len());

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, mask.as_ref())?;
            if layer_indices.contains(&i) {
                captured.push(hidden.clone());
            }
            if i == max_layer {
                break;
            }
        }

        if captured.len() != layer_indices.len() {
            candle_core::bail!(
                "expected {} layer outputs, got {} (max layer index {} >= num_layers {})",
                layer_indices.len(),
                captured.len(),
                max_layer,
                self.layers.len(),
            );
        }

        Tensor::cat(&captured, D::Minus1)
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub struct Qwen3Encoder {
    model: Qwen3TextEncoder,
    tokenizer: tokenizers::Tokenizer,
}

fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn format_prompt_for_flux2(prompt: &str) -> String {
    format!("{}<think>\n\n</think>\n\n", format_prompt_for_qwen3(prompt))
}

impl Qwen3Encoder {
    pub fn load(
        encoder_paths: &[impl AsRef<Path>],
        tokenizer_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cfg = Qwen3Config::flux();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .filter_map(|p| p.as_ref().to_str())
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = Qwen3TextEncoder::new(&cfg, vb)?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        Ok(Self { model, tokenizer })
    }

    /// Encode prompt and extract hidden states from specific layers.
    ///
    /// For Flux.2, `layer_indices` is typically `[9, 18, 27]`, producing
    /// `(1, seq_len, 7680)` embeddings for `context_in_dim = 7680`.
    pub fn encode_with_layers(
        &self,
        prompt: &str,
        device: &Device,
        dtype: DType,
        layer_indices: &[usize],
    ) -> Result<(Tensor, usize)> {
        let formatted = format_prompt_for_flux2(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), device)?;

        let emb = self.model.forward_with_layers(&input_ids, layer_indices)?;
        let emb = emb.to_device(device)?.to_dtype(dtype)?;
        Ok((emb, token_count))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_chat_template() {
        let result = format_prompt_for_qwen3("a cat");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a cat"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
        assert!(!result.contains("<think>"));
    }

    #[test]
    fn flux2_chat_template_includes_thinking() {
        let result = format_prompt_for_flux2("a sunset");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a sunset"));
        assert!(result.contains("<think>\n\n</think>\n\n"));
        assert!(result.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn templates_differ_only_in_thinking_block() {
        let z = format_prompt_for_qwen3("test");
        let f = format_prompt_for_flux2("test");
        assert_eq!(f, format!("{z}<think>\n\n</think>\n\n"));
    }

    #[test]
    fn templates_exact_structure() {
        assert_eq!(
            format_prompt_for_qwen3("hello"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            format_prompt_for_flux2("hello"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn flux_config_dimensions() {
        let cfg = Qwen3Config::flux();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.hidden_size * 3, 7680); // context_in_dim
    }
}
