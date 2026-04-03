//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder, linear_b as linear_bias};

use super::config::Gemma4TextConfig;

// ── RmsNorm (Gemma 4 text — no +1 offset) ──────────────────────────────────
//
// Gemma 4 changed from Gemma 2/3: norm weights are applied directly (no +1
// offset). The stored weights are already the actual scale factors.

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
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

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        for _ in rope_angles..half_dim {
            inv_freq_vec.push(0f32);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
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

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── Trace helpers ──────────────────────────────────────────────────────────

/// Compute RMS norm of a tensor for trace logging. Returns NaN on error.
/// Casts to F32 first to avoid BF16 overflow in the squaring step.
fn trace_rms(xs: &Tensor) -> f32 {
    xs.to_dtype(DType::F32)
        .and_then(|x| x.sqr())
        .and_then(|x| x.mean_all())
        .and_then(|x| x.to_scalar::<f32>())
        .unwrap_or(f32::NAN)
        .sqrt()
}

/// Log top-k token IDs and logit values at trace level.
fn trace_top_k_logits(logits: &Tensor, label: &str) -> Result<()> {
    if tracing::enabled!(tracing::Level::TRACE) {
        let flat = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let flat_vec: Vec<f32> = flat.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = flat_vec.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<_> = indexed
            .iter()
            .take(5)
            .map(|(i, v)| format!("{}:{:.2}", i, v))
            .collect();
        tracing::trace!("{}: [{}]", label, top5.join(", "));
    }
    Ok(())
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

impl KvCache {
    fn reset(&mut self) {
        match self {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }

    fn current_seq_len(&self) -> usize {
        match self {
            KvCache::Normal(c) => c.current_seq_len(),
            KvCache::Rotating(c) => c.current_seq_len(),
        }
    }

    fn k(&self) -> Result<Option<Tensor>> {
        match self {
            KvCache::Normal(c) => c.k(),
            KvCache::Rotating(c) => c.k(),
        }
    }

    fn v(&self) -> Result<Option<Tensor>> {
        match self {
            KvCache::Normal(c) => c.v(),
            KvCache::Rotating(c) => c.v(),
        }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            KvCache::Normal(c) => c.append(k, v),
            KvCache::Rotating(c) => c.append(k, v),
        }
    }
}

pub use crate::shared::types::LayerKvSnapshot;

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    layer_idx: usize,
    /// If set, this layer reads K/V from the donor layer's cache instead of its own.
    kv_shared_layer_index: Option<usize>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let kv_shared_layer_index = cfg.kv_shared_layer_index(layer_idx);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            rotary_emb_global,
            rotary_emb_local,
            layer_idx,
            kv_shared_layer_index,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_caches: &mut [KvCache],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        tracing::trace!(
            "layer {} attn: Q={:?} K={:?} V={:?} ({})",
            self.layer_idx,
            q.shape(),
            k.shape(),
            v.shape(),
            if self.is_sliding { "sliding" } else { "global" },
        );

        // Q/K norms
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        // V norm (RMS without learned weight)
        let v = v_norm(&v, self.rms_norm_eps)?;

        // Apply RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        };

        // KV cache: shared layers read from donor, non-shared append to own cache
        let (k, v) = if let Some(donor_idx) = self.kv_shared_layer_index {
            let donor = &kv_caches[donor_idx];
            let dk = donor
                .k()?
                .ok_or_else(|| candle_core::Error::Msg(format!("donor layer {} K cache empty", donor_idx)))?;
            let dv = donor
                .v()?
                .ok_or_else(|| candle_core::Error::Msg(format!("donor layer {} V cache empty", donor_idx)))?;
            tracing::trace!(
                "layer {} attn: shared, reading KV from donor layer {} (kv_len={})",
                self.layer_idx,
                donor_idx,
                dk.dim(2)?,
            );
            (dk, dv)
        } else {
            let (k, v) = kv_caches[self.layer_idx].append(&k, &v)?;
            tracing::trace!(
                "layer {} attn: appended to own cache (kv_len={})",
                self.layer_idx,
                k.dim(2)?,
            );
            (k, v)
        };

        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // Adjust mask to match actual KV cache length (may differ for shared
        // layers or rotating caches that trimmed to window size).
        let kv_seq_len = k.dim(2)?;
        let mask = mask
            .map(|m| {
                let mask_kv_len = m.dim(D::Minus1)?;
                if mask_kv_len > kv_seq_len {
                    m.narrow(D::Minus1, mask_kv_len - kv_seq_len, kv_seq_len)
                } else {
                    Ok(m.clone())
                }
            })
            .transpose()?;

        tracing::trace!(
            "layer {} attn: kv_seq_len={}, mask={:?}",
            self.layer_idx,
            kv_seq_len,
            mask.as_ref().map(|m| format!("{:?}", m.shape())),
        );

        // Gemma 4 uses QK-norm; attention scaling is 1.0 (not 1/sqrt(head_dim)).
        let scale = 1.0f32;
        let attn_output = if q.device().is_metal() && mask.is_none() {
            // Metal-accelerated SDPA (no mask path — single-token decode)
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?
        } else {
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale as f64)?;
            let attn_weights = match &mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    #[allow(dead_code)]
    is_sliding: bool,
    // Per-Layer Embedding (PLE) injection
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    ple_act_fn: Activation,
    layer_scalar: Option<Tensor>,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        // PLE per-layer weights (only if hidden_size_per_layer_input > 0)
        let ple_dim = cfg.hidden_size_per_layer_input;
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm, layer_scalar) =
            if ple_dim > 0 {
                let gate = candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    ple_dim,
                    vb.pp("per_layer_input_gate"),
                )?;
                let proj = candle_nn::linear_no_bias(
                    ple_dim,
                    cfg.hidden_size,
                    vb.pp("per_layer_projection"),
                )?;
                let norm = RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                )?;
                let scalar = vb.get(1, "layer_scalar")?;
                (Some(gate), Some(proj), Some(norm), Some(scalar))
            } else {
                (None, None, None, None)
            };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            is_sliding,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            ple_act_fn: cfg.hidden_activation,
            layer_scalar,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        per_layer_input: Option<&Tensor>,
        kv_caches: &mut [KvCache],
    ) -> Result<Tensor> {
        // Attention block
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            kv_caches,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("layer {} after attn+res: norm={:.4}", self.self_attn.layer_idx, trace_rms(&xs));
        }

        // MLP block
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("layer {} after mlp+res: norm={:.4}", self.self_attn.layer_idx, trace_rms(&xs));
        }

        // PLE injection (after attention + MLP)
        if let (Some(gate), Some(proj), Some(norm), Some(ple_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = &xs;
            let gated = xs.apply(gate)?.apply(&self.ple_act_fn)?;
            let gated = (gated * ple_input)?;
            let projected = gated.apply(proj)?;
            let normed = norm.forward(&projected)?;
            xs = (residual + normed)?;
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!("layer {} after PLE: norm={:.4}", self.self_attn.layer_idx, trace_rms(&xs));
            }
        }

        // Per-layer scalar
        if let Some(scalar) = &self.layer_scalar {
            xs = xs.broadcast_mul(scalar)?;
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!("layer {} after scalar: norm={:.4}", self.self_attn.layer_idx, trace_rms(&xs));
            }
        }

        Ok(xs)
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    // Per-Layer Embeddings (PLE)
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    ple_dim: usize,
    // Shared KV cache pool — one slot per layer. Shared layers leave their
    // slot empty and read from the donor's slot instead.
    kv_caches: Vec<KvCache>,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        // PLE global weights
        let ple_dim = cfg.hidden_size_per_layer_input;
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                let ple_vocab = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);
                let total_ple_dim = cfg.num_hidden_layers * ple_dim;
                let emb = candle_nn::embedding(
                    ple_vocab,
                    total_ple_dim,
                    vb.pp("embed_tokens_per_layer"),
                )?;
                let proj = candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    total_ple_dim,
                    vb.pp("per_layer_model_projection"),
                )?;
                let norm = RmsNorm::new(
                    ple_dim,
                    cfg.rms_norm_eps,
                    vb.pp("per_layer_projection_norm"),
                )?;
                (Some(emb), Some(proj), Some(norm))
            } else {
                (None, None, None)
            };

        // Build KV cache pool — one per layer. Shared layers' slots are never
        // written to; they read from their donor's slot during forward.
        let mut kv_caches = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            if cfg.is_sliding(layer_idx) {
                kv_caches.push(KvCache::Rotating(
                    candle_nn::kv_cache::RotatingKvCache::new(2, cfg.effective_sliding_window()),
                ));
            } else {
                kv_caches.push(KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                    2,
                    cfg.max_position_embeddings,
                )));
            }
        }

        if cfg.num_kv_shared_layers > 0 {
            tracing::info!(
                "KV sharing: layers {}-{} share caches with donors in 0-{}",
                cfg.first_kv_shared_layer_idx(),
                cfg.num_hidden_layers - 1,
                cfg.first_kv_shared_layer_idx() - 1,
            );
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            ple_dim,
            kv_caches,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = (xs * (self.hidden_size as f64).sqrt())?;
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                "embed_tokens: shape={:?}, dtype={:?}, norm={:.4}",
                xs.shape(),
                xs.dtype(),
                trace_rms(&xs),
            );
        }
        Ok(xs)
    }

    /// Compute per-layer embedding inputs from token IDs and main embeddings.
    /// Returns a tensor of shape [B, S, num_layers, ple_dim] or None if PLE is disabled.
    pub fn compute_per_layer_inputs(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (emb_pl, proj, norm) = match (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(e), Some(p), Some(n)) => (e, p, n),
            _ => return Ok(None),
        };

        let dims = input_ids.dims();
        let ple_embed_scale = (self.ple_dim as f64).sqrt();

        // Token-identity stream: embed_tokens_per_layer(input_ids) * sqrt(ple_dim)
        // -> reshape [B, S, num_layers, ple_dim]
        let token_identity = (emb_pl.forward(input_ids)? * ple_embed_scale)?;
        let mut shape = dims.to_vec();
        shape.push(self.layers.len());
        shape.push(self.ple_dim);
        let token_identity = token_identity.reshape(shape.as_slice())?;
        tracing::trace!("PLE token_identity shape={:?}", token_identity.shape());

        // Context-aware stream: per_layer_model_projection(inputs_embeds) * 1/sqrt(hidden_size)
        // -> reshape [B, S, num_layers, ple_dim] -> RMSNorm
        let proj_scale = (self.hidden_size as f64).powf(-0.5);
        let context_proj = (inputs_embeds.apply(proj)? * proj_scale)?;
        let context_proj = context_proj.reshape(shape.as_slice())?;
        let context_proj = norm.forward(&context_proj)?;
        tracing::trace!("PLE context_proj shape={:?}", context_proj.shape());

        // Combine: (context_proj + token_identity) * 1/sqrt(2)
        let combined = ((context_proj + token_identity)? * std::f64::consts::FRAC_1_SQRT_2)?;
        tracing::trace!("PLE combined shape={:?}", combined.shape());

        Ok(Some(combined))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;
        self.forward_embeds(&xs, seqlen_offset, b_size, seq_len, per_layer_inputs.as_ref())
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
        per_layer_inputs: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        // Destructure to borrow layers and kv_caches separately.
        let Self {
            layers, kv_caches, ..
        } = self;

        let mut xs = xs.clone();
        for (i, layer) in layers.iter().enumerate() {
            // Extract this layer's PLE slice: [B, S, ple_dim]
            let ple_slice = per_layer_inputs
                .map(|pli| pli.narrow(pli.dims().len() - 2, i, 1)?.squeeze(pli.dims().len() - 2))
                .transpose()?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                ple_slice.as_ref(),
                kv_caches,
            )?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        tracing::trace!("logits shape={:?}, dtype={:?}", logits.shape(), logits.dtype());
        match self.final_logit_softcapping {
            None => {
                if tracing::enabled!(tracing::Level::TRACE) {
                    trace_top_k_logits(&logits, "top-5 logits")?;
                }
                Ok(logits)
            }
            Some(sc) => {
                let orig_dtype = logits.dtype();
                let logits = logits.to_dtype(DType::F32)?;
                let logits = ((logits / sc)?.tanh()? * sc)?.to_dtype(orig_dtype)?;
                if tracing::enabled!(tracing::Level::TRACE) {
                    trace_top_k_logits(&logits, "top-5 logits (softcapped)")?;
                }
                Ok(logits)
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }

    /// Current number of cached tokens (from the first layer's cache).
    pub fn kv_cache_seq_len(&self) -> usize {
        self.kv_caches[0].current_seq_len()
    }

    /// Extract K/V tensors from all layers for saving.
    pub fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>> {
        self.kv_caches
            .iter()
            .enumerate()
            .map(|(idx, cache)| {
                let is_sliding = matches!(cache, KvCache::Rotating(_));
                let offset = match cache {
                    KvCache::Rotating(c) => c.offset(),
                    _ => 0,
                };
                Ok(LayerKvSnapshot {
                    layer_idx: idx,
                    is_sliding,
                    k_data: cache.k()?,
                    v_data: cache.v()?,
                    offset,
                    current_seq_len: cache.current_seq_len(),
                })
            })
            .collect()
    }

    /// Restore K/V cache from saved snapshots.
    pub fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()> {
        for snap in snapshots {
            if snap.layer_idx < self.kv_caches.len() {
                let cache = &mut self.kv_caches[snap.layer_idx];
                cache.reset();
                if let (Some(k), Some(v)) = (&snap.k_data, &snap.v_data) {
                    match cache {
                        KvCache::Normal(c) => {
                            c.k_cache_mut().append(k)?;
                            c.v_cache_mut().append(v)?;
                        }
                        KvCache::Rotating(c) => {
                            c.k_cache_mut().append(k)?;
                            c.v_cache_mut().append(v)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
