#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::{self, concatenate, indexing::IndexOp};
use mlx_rs::{Array, Dtype, array};

use crate::mlx_helpers::kv_cache::KvCache;
use crate::mlx_helpers::weight_loader::{get_weight, get_weight_as};
use crate::model_config::TextConfig;

use super::linear_attn::GatedDeltaNet;

// ---------------------------------------------------------------------------
// Rotary Embedding (partial RoPE)
// ---------------------------------------------------------------------------

pub struct RotaryEmbedding {
    cos: Array,
    sin: Array,
    rope_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(cfg: &TextConfig) -> anyhow::Result<Self> {
        let rope_dim = cfg.rope_dim();
        let theta = cfg.rope_parameters.rope_theta as f32;

        let inv_freq: Vec<f32> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / rope_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Array::from_slice(&inv_freq, &[1, inv_freq_len as i32]);

        let max_seq = cfg.max_position_embeddings.min(8192);
        let t: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
        let t = Array::from_slice(&t, &[max_seq as i32, 1]);
        let freqs = ops::matmul(&t, &inv_freq)?;
        let freqs = concatenate(&[&freqs, &freqs], -1)?;
        let sin = ops::sin(&freqs)?;
        let cos = ops::cos(&freqs)?;

        Ok(Self {
            cos,
            sin,
            rope_dim,
        })
    }

    pub fn forward(
        &self,
        q: &Array,
        k: &Array,
        seqlen_offset: usize,
    ) -> Result<(Array, Array), Exception> {
        let shape = q.shape();
        let seq_len = shape[2] as usize;
        let head_dim = shape[3] as usize;
        let rope_dim = self.rope_dim;

        if rope_dim >= head_dim {
            let cos = self.cos.index((seqlen_offset as i32..(seqlen_offset + seq_len) as i32, ..));
            let sin = self.sin.index((seqlen_offset as i32..(seqlen_offset + seq_len) as i32, ..));
            let q_rot = apply_rope(q, &cos, &sin)?;
            let k_rot = apply_rope(k, &cos, &sin)?;
            return Ok((q_rot, k_rot));
        }

        // Partial rotary: only first rope_dim dimensions
        let q_rot = q.index((.., .., .., ..rope_dim as i32));
        let q_pass = q.index((.., .., .., rope_dim as i32..));
        let k_rot = k.index((.., .., .., ..rope_dim as i32));
        let k_pass = k.index((.., .., .., rope_dim as i32..));

        let cos = self.cos.index((seqlen_offset as i32..(seqlen_offset + seq_len) as i32, ..));
        let sin = self.sin.index((seqlen_offset as i32..(seqlen_offset + seq_len) as i32, ..));

        let q_rot = apply_rope(&q_rot, &cos, &sin)?;
        let k_rot = apply_rope(&k_rot, &cos, &sin)?;

        let q_out = concatenate(&[&q_rot, &q_pass], -1)?;
        let k_out = concatenate(&[&k_rot, &k_pass], -1)?;
        Ok((q_out, k_out))
    }
}

/// Apply rotary embeddings: x * cos + rotate_half(x) * sin
fn apply_rope(xs: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let shape = xs.shape();
    let d = *shape.last().unwrap() as usize;
    let half = (d / 2) as i32;
    let x1 = xs.index((.., .., .., ..half));
    let x2 = xs.index((.., .., .., half..));
    // cos/sin shape: (seq, rope_dim) -> expand to (1, 1, seq, rope_dim)
    let cos = ops::expand_dims(cos, &[0, 1])?;
    let sin = ops::expand_dims(sin, &[0, 1])?;
    let rotated = concatenate(&[&ops::negative(&x2)?, &x1], -1)?;
    let result = xs.multiply(cos)?.add(rotated.multiply(sin)?)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

pub struct Mlp {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl Mlp {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let h = hidden_size as i32;
        let i = intermediate_size as i32;
        let mut gate_proj = nn::LinearBuilder::new(h, i).bias(false).build()?;
        let mut up_proj = nn::LinearBuilder::new(h, i).bias(false).build()?;
        let mut down_proj = nn::LinearBuilder::new(i, h).bias(false).build()?;

        gate_proj.weight = get_weight(weights, &format!("{prefix}.gate_proj.weight"))?;
        up_proj.weight = get_weight(weights, &format!("{prefix}.up_proj.weight"))?;
        down_proj.weight = get_weight(weights, &format!("{prefix}.down_proj.weight"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, xs: &Array) -> Result<Array, Exception> {
        let gate = nn::silu(self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&gate.multiply(up)?)
    }
}

// ---------------------------------------------------------------------------
// Full Attention Layer
// ---------------------------------------------------------------------------

pub struct FullAttentionLayer {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    mlp: Mlp,
    kv_cache: KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    scale: f32,
}

impl FullAttentionLayer {
    pub fn new(
        cfg: &TextConfig,
        layer_idx: usize,
        weights: &HashMap<String, Array>,
    ) -> anyhow::Result<Self> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let attn_prefix = format!("{prefix}.self_attn");
        let h = cfg.hidden_size as i32;
        let head_dim = cfg.head_dim as i32;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        // Q projection is 2x (query + output gate)
        let mut q_proj = nn::LinearBuilder::new(h, (num_heads as i32) * head_dim * 2)
            .bias(false)
            .build()?;
        let mut k_proj = nn::LinearBuilder::new(h, (num_kv_heads as i32) * head_dim)
            .bias(false)
            .build()?;
        let mut v_proj = nn::LinearBuilder::new(h, (num_kv_heads as i32) * head_dim)
            .bias(false)
            .build()?;
        let mut o_proj = nn::LinearBuilder::new((num_heads as i32) * head_dim, h)
            .bias(false)
            .build()?;

        q_proj.weight = get_weight(weights, &format!("{attn_prefix}.q_proj.weight"))?;
        k_proj.weight = get_weight(weights, &format!("{attn_prefix}.k_proj.weight"))?;
        v_proj.weight = get_weight(weights, &format!("{attn_prefix}.v_proj.weight"))?;
        o_proj.weight = get_weight(weights, &format!("{attn_prefix}.o_proj.weight"))?;

        let mut q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;
        let mut k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;

        q_norm.weight = get_weight(weights, &format!("{attn_prefix}.q_norm.weight"))?;
        k_norm.weight = get_weight(weights, &format!("{attn_prefix}.k_norm.weight"))?;

        let mut input_layernorm = nn::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;

        input_layernorm.weight =
            get_weight(weights, &format!("{prefix}.input_layernorm.weight"))?;
        post_attention_layernorm.weight =
            get_weight(weights, &format!("{prefix}.post_attention_layernorm.weight"))?;

        let mlp = Mlp::new(
            cfg.hidden_size,
            cfg.mlp_intermediate_size(),
            weights,
            &format!("{prefix}.mlp"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            kv_cache: KvCache::new(),
            num_heads,
            num_kv_heads,
            head_dim: cfg.head_dim,
            n_kv_groups: num_heads / num_kv_heads,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &mut self,
        xs: &Array,
        mask: Option<&Array>,
        rotary: &RotaryEmbedding,
        seqlen_offset: usize,
    ) -> Result<Array, Exception> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let shape = xs.shape();
        let b_sz = shape[0];
        let q_len = shape[1];
        let h = self.num_heads as i32;
        let kv_h = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        // Q projection: output is 2x head_dim (query + gate)
        let q_full = self.q_proj.forward(&xs)?;
        let q_full = q_full.reshape(&[b_sz, q_len, h, hd * 2])?;
        let q = q_full.index((.., .., .., ..hd));
        let gate = q_full.index((.., .., .., hd..));
        let gate = gate.reshape(&[b_sz, q_len, h * hd])?;

        let k = self.k_proj.forward(&xs)?;
        let v = self.v_proj.forward(&xs)?;

        // (b, seq, heads, head_dim) -> (b, heads, seq, head_dim)
        let q = q.reshape(&[b_sz, q_len, h, hd])?.transpose(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b_sz, q_len, kv_h, hd])?.transpose(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b_sz, q_len, kv_h, hd])?.transpose(&[0, 2, 1, 3])?;

        // Per-head QK norms
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Partial RoPE
        let (q, k) = rotary.forward(&q, &k, seqlen_offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads
        let (k, v) = if self.n_kv_groups > 1 {
            (
                repeat_kv(&k, self.n_kv_groups)?,
                repeat_kv(&v, self.n_kv_groups)?,
            )
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let attn_weights = ops::matmul(&q, &k.transpose(&[0, 1, 3, 2])?)?
            .multiply(array!(self.scale))?;
        let attn_weights = match mask {
            Some(m) => attn_weights.add(m)?,
            None => attn_weights,
        };
        let attn_weights = ops::softmax(&attn_weights, &[-1])?;
        let mut attn_output = ops::matmul(&attn_weights, &v)?;

        // (b, heads, seq, hd) -> (b, seq, heads*hd)
        attn_output = attn_output
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[b_sz, q_len, h * hd])?;

        // Output gating: attn * sigmoid(gate)
        let gate = ops::sigmoid(&gate)?;
        attn_output = attn_output.multiply(gate)?;

        let xs = self.o_proj.forward(&attn_output)?;
        let xs = xs.add(residual)?;

        // MLP
        let residual = &xs;
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual.add(mlp_out)
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }
}

fn repeat_kv(xs: &Array, n_rep: usize) -> Result<Array, Exception> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let shape = xs.shape();
    let (b, n_kv, s, hd) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded = ops::expand_dims(xs, &[2])?;
    let expanded = ops::broadcast_to(&expanded, &[b, n_kv, n_rep as i32, s, hd])?;
    expanded.reshape(&[b, n_kv * n_rep as i32, s, hd])
}

// ---------------------------------------------------------------------------
// Linear Attention Layer (wraps GatedDeltaNet)
// ---------------------------------------------------------------------------

pub struct LinearAttentionLayer {
    delta_net: GatedDeltaNet,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    mlp: Mlp,
}

impl LinearAttentionLayer {
    pub fn new(
        cfg: &TextConfig,
        layer_idx: usize,
        weights: &HashMap<String, Array>,
    ) -> anyhow::Result<Self> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let h = cfg.hidden_size as i32;

        let delta_net = GatedDeltaNet::new(cfg, weights, &format!("{prefix}.linear_attn"))?;

        let mut input_layernorm = nn::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;

        input_layernorm.weight =
            get_weight(weights, &format!("{prefix}.input_layernorm.weight"))?;
        post_attention_layernorm.weight =
            get_weight(weights, &format!("{prefix}.post_attention_layernorm.weight"))?;

        let mlp = Mlp::new(
            cfg.hidden_size,
            cfg.mlp_intermediate_size(),
            weights,
            &format!("{prefix}.mlp"),
        )?;

        Ok(Self {
            delta_net,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(&mut self, xs: &Array) -> Result<Array, Exception> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.delta_net.forward(&xs)?;
        let xs = xs.add(residual)?;

        let residual = &xs;
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual.add(mlp_out)
    }

    pub fn reset(&mut self) {
        self.delta_net.reset_state();
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer (enum dispatching full vs linear attention)
// ---------------------------------------------------------------------------

pub enum DecoderLayer {
    Full(FullAttentionLayer),
    Linear(LinearAttentionLayer),
}

impl DecoderLayer {
    pub fn forward(
        &mut self,
        xs: &Array,
        mask: Option<&Array>,
        rotary: &RotaryEmbedding,
        seqlen_offset: usize,
    ) -> Result<Array, Exception> {
        match self {
            Self::Full(layer) => layer.forward(xs, mask, rotary, seqlen_offset),
            Self::Linear(layer) => layer.forward(xs),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Full(layer) => layer.reset(),
            Self::Linear(layer) => layer.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 Text Model
// ---------------------------------------------------------------------------

pub struct Qwen35TextModel {
    embed_tokens: nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: nn::Linear,
    rotary_emb: RotaryEmbedding,
    pub num_attn_heads: usize,
}

impl Qwen35TextModel {
    pub fn new(cfg: &TextConfig, weights: &HashMap<String, Array>) -> anyhow::Result<Self> {
        let h = cfg.hidden_size as i32;
        let v = cfg.vocab_size as i32;

        // Embedding
        let mut embed_tokens = nn::Embedding::new(v, h)?;
        embed_tokens.weight =
            get_weight(weights, "model.language_model.embed_tokens.weight")?;

        // Rotary embedding
        let rotary_emb = RotaryEmbedding::new(cfg)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            if cfg.is_linear_attention_layer(i) {
                layers.push(DecoderLayer::Linear(LinearAttentionLayer::new(
                    cfg, i, weights,
                )?));
            } else {
                layers.push(DecoderLayer::Full(FullAttentionLayer::new(
                    cfg, i, weights,
                )?));
            }
            tracing::debug!(layer = i, "loaded layer");
        }

        // Final norm
        let mut norm = nn::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps as f32)
            .build()?;
        norm.weight = get_weight(weights, "model.language_model.norm.weight")?;

        // LM head
        let mut lm_head = nn::LinearBuilder::new(h, v).bias(false).build()?;
        if !cfg.tie_word_embeddings {
            lm_head.weight = get_weight(weights, "lm_head.weight")?;
        } else {
            lm_head.weight = embed_tokens.weight.clone();
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            num_attn_heads: cfg.num_attention_heads,
        })
    }

    pub fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_embeds(
        &mut self,
        mut xs: Array,
        mask: Option<&Array>,
        seqlen_offset: usize,
    ) -> Result<Array, Exception> {
        let seq_len = xs.shape()[1];

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let t = std::time::Instant::now();
            xs = layer.forward(&xs, mask, &self.rotary_emb, seqlen_offset)?;
            tracing::debug!(
                layer = i,
                kind = match layer {
                    DecoderLayer::Full(_) => "full",
                    DecoderLayer::Linear(_) => "linear",
                },
                elapsed_ms = t.elapsed().as_millis() as u64,
                "layer done"
            );
        }

        xs = self.norm.forward(&xs)?;
        // Only take logits for the last token
        let last = xs.index((.., (seq_len - 1).., ..));
        self.lm_head.forward(&last)
    }

    pub fn reset_caches(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
