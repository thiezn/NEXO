#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used
)]

use std::sync::{Arc, Mutex};

use local_inference_helpers::candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use local_inference_helpers::candle_nn::{
    embedding, kv_cache::KvCache, linear, linear_b, rms_norm, Activation, Embedding, Linear,
    Module, RmsNorm, VarBuilder,
};

use super::config::TextConfig;
use super::linear_attn::GatedDeltaNet;
use super::moe::MoeMlp;

/// Standard rotary embedding (partial RoPE for Qwen3.5)
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rope_dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        let rope_dim = cfg.rope_dim();
        let theta = cfg.rope_parameters.rope_theta as f32;

        let inv_freq: Vec<_> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / rope_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let max_seq = cfg.max_position_embeddings.min(8192); // precompute a reasonable amount
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            rope_dim,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let rope_dim = self.rope_dim;

        if rope_dim >= head_dim {
            // Full rotary
            let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
            let q_rot = rope(q, &cos, &sin)?;
            let k_rot = rope(k, &cos, &sin)?;
            return Ok((q_rot, k_rot));
        }

        // Partial rotary: only first rope_dim dimensions get rotary, rest pass through
        let q_rot = q.narrow(D::Minus1, 0, rope_dim)?;
        let q_pass = q.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, rope_dim)?;
        let k_pass = k.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?;

        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        let q_rot = rope(&q_rot, &cos, &sin)?;
        let k_rot = rope(&k_rot, &cos, &sin)?;

        let q_out = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
        let k_out = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
        Ok((q_out, k_out))
    }
}

/// Apply rotary embeddings to a tensor (batch, heads, seq, dim)
fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = xs.dims4()?;
    let half = d / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    // cos/sin shape: (seq, half) → need (1, 1, seq, half)
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    xs.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

/// Standard SwiGLU MLP (for dense models / full attention layers)
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, act: Activation, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_b(hidden_size, intermediate_size, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_size, intermediate_size, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_size, hidden_size, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

/// MLP wrapper that dispatches to dense or MoE
enum MlpLayer {
    Dense(Mlp),
    Moe(MoeMlp),
}

impl MlpLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

/// Full attention layer (standard QKV attention with RoPE)
struct FullAttentionLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    softmax_scale: f64,
    kv_cache: Arc<Mutex<KvCache>>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: MlpLayer,
}

impl FullAttentionLayer {
    fn new(
        cfg: &TextConfig,
        rotary_emb: &Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let _ = rotary_emb; // rotary is shared, applied in forward
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        let attn_vb = vb.pp("self_attn");
        let q_proj = linear_b(hidden_sz, num_heads * cfg.head_dim, false, attn_vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_sz, num_kv_heads * cfg.head_dim, false, attn_vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_sz, num_kv_heads * cfg.head_dim, false, attn_vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * cfg.head_dim, hidden_sz, false, attn_vb.pp("o_proj"))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn_vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn_vb.pp("k_norm"))?;

        let input_layernorm = rms_norm(hidden_sz, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(hidden_sz, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let mlp = if cfg.is_moe() {
            MlpLayer::Moe(MoeMlp::new(cfg, vb.pp("mlp"))?)
        } else {
            MlpLayer::Dense(Mlp::new(hidden_sz, cfg.mlp_intermediate_size(), cfg.hidden_act, vb.pp("mlp"))?)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim: cfg.head_dim,
            n_kv_groups: num_heads / num_kv_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: Arc::new(Mutex::new(KvCache::new(2, 8192))),
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(&xs)?;
        let mut k = self.k_proj.forward(&xs)?;
        let mut v = self.v_proj.forward(&xs)?;

        q = q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        k = k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        v = v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        let (q, k) = rotary.forward(&q, &k, seqlen_offset)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let (k, v) = self
            .kv_cache
            .lock()
            .expect("kv cache lock")
            .append(&k, &v)?;

        let k = repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = local_inference_helpers::candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let mut attn_output = attn_weights.matmul(&v)?;

        attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;
        let xs = self.o_proj.forward(&attn_output)?;
        let xs = (xs + residual)?;

        // MLP
        let residual = &xs;
        let mlp_out = self.mlp.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + mlp_out
    }
}

/// Linear attention layer (GatedDeltaNet + MLP)
struct LinearAttentionLayer {
    delta_net: GatedDeltaNet,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: MlpLayer,
}

impl LinearAttentionLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let delta_net = GatedDeltaNet::new(cfg, vb.pp("self_attn"))?;
        let input_layernorm = rms_norm(hidden_sz, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(hidden_sz, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let mlp = if cfg.is_moe() {
            MlpLayer::Moe(MoeMlp::new(cfg, vb.pp("mlp"))?)
        } else {
            MlpLayer::Dense(Mlp::new(hidden_sz, cfg.mlp_intermediate_size(), cfg.hidden_act, vb.pp("mlp"))?)
        };

        Ok(Self {
            delta_net,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.delta_net.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let mlp_out = self.mlp.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + mlp_out
    }
}

enum DecoderLayer {
    Full(FullAttentionLayer),
    Linear(LinearAttentionLayer),
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Full(layer) => layer.forward(xs, attention_mask, rotary, seqlen_offset),
            Self::Linear(layer) => layer.forward(xs),
        }
    }
}

pub struct Qwen35TextModel {
    embed_tokens: Embedding,
    pub(super) norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    lm_head: Linear,
    rotary_emb: Arc<RotaryEmbedding>,
    pub(super) dtype: DType,
    pub(super) num_attn_heads: usize,
}

impl Qwen35TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, vb.device(), vb_m.dtype())?);

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb_l.pp(layer_idx);
            if cfg.is_linear_attention_layer(layer_idx) {
                layers.push(DecoderLayer::Linear(LinearAttentionLayer::new(
                    cfg, layer_vb,
                )?));
            } else {
                layers.push(DecoderLayer::Full(FullAttentionLayer::new(
                    cfg, &rotary_emb, layer_vb,
                )?));
            }
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            rotary_emb,
            dtype: vb.dtype(),
            num_attn_heads: cfg.num_attention_heads,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_embeds(
        &mut self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;

        for layer in &mut self.layers {
            xs = layer.forward(&xs, attention_mask, &self.rotary_emb, seqlen_offset)?;
        }

        xs = xs.apply(&self.norm)?;
        self.lm_head
            .forward(&xs)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    pub fn reset_kv_caches(&mut self) {
        for layer in &mut self.layers {
            match layer {
                DecoderLayer::Full(l) => {
                    *l.kv_cache.lock().expect("kv lock") = KvCache::new(2, 8192);
                }
                DecoderLayer::Linear(l) => {
                    l.delta_net.reset_state();
                }
            }
        }
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
