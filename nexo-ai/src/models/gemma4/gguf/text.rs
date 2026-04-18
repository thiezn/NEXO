//! Quantized Gemma 4 text decoder (GGUF).
//!
//! Implements the Gemma 4 architecture with quantized weights loaded from GGUF files.
//! Based on candle's quantized_gemma3.rs but extended for Gemma 4 features:
//! - Dual head dimensions (sliding vs global attention)
//! - V-norm (unweighted RMS)
//! - Proportional RoPE for global layers
//! - KV sharing for tail layers
//! - Per-Layer Embeddings (PLE, optional)
//! - Four norms per decoder layer

use candle_core::quantized::QTensor;
use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};

use crate::models::gemma4::generation::TextForward;
use crate::shared::types::LayerKvSnapshot;

pub const MAX_SEQ_LEN: usize = 131072;
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

// ── Quantized RmsNorm ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

// ── QMatMul wrapper ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        // Gemma 4 uses GELU PyTorch Tanh approximation (not gelu_erf)
        let activated = Activation::GeluPytorchTanh.forward(&gate)?;
        let gated = (activated * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

// ── Rotary Embedding (standard, for sliding layers) ───────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn apply_rotary_emb_q(&self, q: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
    }
}

// ── Proportional Rotary Embedding (for global layers) ─────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        head_dim: usize,
        rope_frequency: f32,
        partial_rotary_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / rope_frequency.powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        for _ in rope_angles..half_dim {
            inv_freq_vec.push(0f32);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), device)?;
        let t = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn apply_rotary_emb_q(&self, q: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
    }
}

// ── Layer weights ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention projections — K/V optional for shared layers
    attention_wq: QMatMul,
    attention_wk: Option<QMatMul>,
    attention_wv: Option<QMatMul>,
    attention_wo: QMatMul,

    // Q/K norms — K norm optional for shared layers
    attention_q_norm: RmsNorm,
    attention_k_norm: Option<RmsNorm>,

    // V-norm: unweighted RMS norm applied to V (ones weight)
    v_norm_weight: Option<Tensor>,
    v_norm_eps: f32,

    // Four norms per decoder layer
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    // Feed-forward
    mlp: Mlp,

    // Attention parameters
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,

    is_sliding: bool,
    sliding_window_size: Option<usize>,

    // RoPE — one of these is used depending on is_sliding
    rotary_emb_local: RotaryEmbedding,
    rotary_emb_global: ProportionalRotaryEmbedding,

    // KV sharing: if set, reads K/V from this donor layer's cache
    kv_shared_layer_index: Option<usize>,

    // PLE per-layer weights
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_scalar: Option<Tensor>,

    neg_inf: Tensor,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            // Cached positions: for global attention all are visible,
            // for sliding window only those within the window.
            let mask0: Vec<u32> = if let Some(sw) = self.sliding_window_size {
                (0..seq_len)
                    .flat_map(|i| {
                        let query_pos = index_pos + i;
                        (0..index_pos).map(move |j| {
                            if query_pos.saturating_sub(j) <= sw {
                                1u32
                            } else {
                                0u32
                            }
                        })
                    })
                    .collect()
            } else {
                vec![1u32; seq_len * index_pos]
            };
            let mask0 = Tensor::from_slice(&mask0, (seq_len, index_pos), device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(DType::F32)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        all_kv_caches: &mut [Option<(Tensor, Tensor)>],
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        if tracing::enabled!(tracing::Level::TRACE) && layer_idx == 0 && index_pos == 0 {
            tracing::trace!(
                "attn layer 0: x=({},{},{}), n_head={}, n_kv_head={}, head_dim={}, shared={}, sliding={}",
                b_sz,
                seq_len,
                n_embd,
                self.n_head,
                self.n_kv_head,
                self.head_dim,
                self.kv_shared_layer_index.is_some(),
                self.is_sliding,
            );
        }

        let q = self.attention_wq.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.attention_q_norm.forward(&q)?;

        // Branch: shared layers skip K/V entirely, non-shared compute + cache
        let (q, k, v) = if let Some(donor_idx) = self.kv_shared_layer_index {
            let q = if self.is_sliding {
                self.rotary_emb_local.apply_rotary_emb_q(&q, index_pos)?
            } else {
                self.rotary_emb_global.apply_rotary_emb_q(&q, index_pos)?
            };
            let (dk, dv) = all_kv_caches[donor_idx]
                .as_ref()
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("donor layer {} K/V cache empty", donor_idx))
                })?
                .clone();
            (q, dk, dv)
        } else {
            let k = self.attention_wk.as_ref().unwrap().forward(x)?;
            let v = self.attention_wv.as_ref().unwrap().forward(x)?;

            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;

            // Q/K norms
            let k = self.attention_k_norm.as_ref().unwrap().forward(&k)?;

            // V-norm: unweighted RMS
            let v = if let Some(v_norm_w) = &self.v_norm_weight {
                candle_nn::ops::rms_norm(&v.contiguous()?, v_norm_w, self.v_norm_eps)?
            } else {
                v
            };

            // RoPE
            let (q, k) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &k, index_pos)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &k, index_pos)?
            };

            // KV cache
            let (k, v) = match &all_kv_caches[layer_idx] {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[k_cache, &k], 2)?;
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                }
            };
            all_kv_caches[layer_idx] = Some((k.clone(), v.clone()));
            (q, k, v)
        };

        // GQA: repeat KV heads
        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled Dot-Product Attention
        // Gemma 4 uses QKV-norm; attention scale is 1.0 (matching safetensors model)
        let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0.0f64)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        self.attention_wo.forward(&attn_output)
    }
}

// ── QuantizedTextModel ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuantizedTextModel {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
    // PLE global weights
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<QMatMul>,
    per_layer_projection_norm: Option<RmsNorm>,
    ple_dim: usize,
    num_layers: usize,
    final_logit_softcapping: Option<f64>,
}

impl QuantizedTextModel {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Detect architecture prefix
        let prefix = ["gemma4", "gemma3", "gemma2", "gemma"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma4");

        tracing::info!("GGUF architecture prefix: {prefix}");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        // Try to read partial_rotary_factor for global layers (Gemma 4 specific)
        let partial_rotary_factor = md_get("rope.partial_rotary_factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(0.25) as f64;

        // Per-attention-type RoPE dimensions
        // rope.dimension_count_swa = sliding head dim (e.g. 256)
        // rope.dimension_count = global head dim (e.g. 512)
        let sliding_head_dim = md_get("rope.dimension_count_swa")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(key_length);
        let global_head_dim = md_get("rope.dimension_count")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(key_length);

        tracing::info!(
            "GGUF config: blocks={}, heads={}, kv_heads={}, global_head_dim={}, sliding_head_dim={}, embed={}, \
             sliding_window={}, sliding_type={}, partial_rotary={}",
            block_count,
            head_count,
            head_count_kv,
            global_head_dim,
            sliding_head_dim,
            embedding_length,
            sliding_window_size,
            sliding_window_type,
            partial_rotary_factor,
        );

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;

        // Final norm
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
            device,
        )?;

        // Output head (may be tied to embeddings)
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };

        // Build per-type RoPE with correct head dimensions
        let rotary_emb_local =
            RotaryEmbedding::new(sliding_head_dim, rope_freq_base_sliding, device)?;
        let rotary_emb_global = ProportionalRotaryEmbedding::new(
            global_head_dim,
            rope_freq_base,
            partial_rotary_factor,
            device,
        )?;

        // Read sliding_window_pattern from GGUF metadata (explicit per-layer array)
        let sliding_pattern: Vec<bool> = {
            let key = format!("{prefix}.attention.sliding_window_pattern");
            match ct.metadata.get(&key) {
                Some(gguf_file::Value::Array(arr)) => arr
                    .iter()
                    .map(|v| matches!(v, gguf_file::Value::Bool(true)))
                    .collect(),
                _ => {
                    // Fallback: compute from sliding_window_type
                    (0..block_count)
                        .map(|i| (i + 1) % sliding_window_type > 0)
                        .collect()
                }
            }
        };

        // KV sharing: last N layers share K/V caches with donor layers
        let shared_kv_layers = md_get("attention.shared_kv_layers")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(0);
        let first_shared = block_count.saturating_sub(shared_kv_layers);

        // Logit softcapping
        let final_logit_softcapping = md_get("final_logit_softcapping")
            .and_then(|m| m.to_f32())
            .ok()
            .map(|v| v as f64);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let blk = format!("blk.{layer_idx}");

            let is_sliding = sliding_pattern.get(layer_idx).copied().unwrap_or(true);
            let sliding_window = is_sliding.then_some(sliding_window_size);

            // KV sharing: layers >= first_shared share K/V from a donor layer
            let is_shared = layer_idx >= first_shared;

            // Head dim per layer type: sliding uses smaller dim, global uses larger
            let layer_head_dim = if is_sliding {
                sliding_head_dim
            } else {
                global_head_dim
            };

            // Attention projections
            let attention_wq = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{blk}.attn_q.weight"),
                device,
            )?)?;

            let (attention_wk, attention_wv, attention_k_norm, v_norm_weight) = if is_shared {
                (None, None, None, None)
            } else {
                let wk = QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{blk}.attn_k.weight"),
                    device,
                )?)?;
                let wv = QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{blk}.attn_v.weight"),
                    device,
                )?)?;
                let kn = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{blk}.attn_k_norm.weight"), device)?,
                    rms_norm_eps,
                    device,
                )?;
                // V-norm: identity weight (ones) for unweighted RMS norm
                let vw = Tensor::ones(layer_head_dim, DType::F32, device)?;
                (Some(wk), Some(wv), Some(kn), Some(vw))
            };

            let attention_wo = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{blk}.attn_output.weight"),
                device,
            )?)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;

            // Four norms per layer
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk}.attn_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk}.ffn_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{blk}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;

            // MLP
            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{blk}.ffn_gate.weight"),
                    device,
                )?)?,
                feed_forward_up: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{blk}.ffn_up.weight"),
                    device,
                )?)?,
                feed_forward_down: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{blk}.ffn_down.weight"),
                    device,
                )?)?,
            };

            // PLE per-layer weights
            let (
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                layer_scalar,
            ) = match ct.tensor(reader, &format!("{blk}.inp_gate.weight"), device) {
                Ok(gate_qt) => {
                    let gate = QMatMul::from_qtensor(gate_qt)?;
                    let proj = QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{blk}.proj.weight"),
                        device,
                    )?)?;
                    let norm = RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{blk}.post_norm.weight"), device)?,
                        rms_norm_eps,
                        device,
                    )?;
                    let scalar = ct
                        .tensor(reader, &format!("{blk}.layer_output_scale.weight"), device)?
                        .dequantize(device)?;
                    (Some(gate), Some(proj), Some(norm), Some(scalar))
                }
                Err(_) => (None, None, None, None),
            };

            // Determine KV donor for shared layers — search only non-shared layers
            // with the same attention type (sliding vs global)
            let kv_shared_layer_index = if is_shared {
                let target_is_sliding = is_sliding;
                sliding_pattern[..first_shared]
                    .iter()
                    .rposition(|&is_sl| is_sl == target_is_sliding)
            } else {
                None
            };

            if is_shared {
                tracing::debug!(
                    "layer {}: shared (donor={:?}, {})",
                    layer_idx,
                    kv_shared_layer_index,
                    if is_sliding { "sliding" } else { "global" },
                );
            }

            layers.push(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_wo,
                attention_q_norm,
                attention_k_norm,
                v_norm_weight,
                v_norm_eps: rms_norm_eps as f32,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: layer_head_dim,
                is_sliding,
                sliding_window_size: sliding_window,
                rotary_emb_local: rotary_emb_local.clone(),
                rotary_emb_global: rotary_emb_global.clone(),
                kv_shared_layer_index,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                layer_scalar,
                neg_inf: neg_inf.clone(),
            });
        }

        let kv_caches = vec![None; block_count];

        // PLE global weights
        let ple_dim = md_get("embedding_length_per_layer_input")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(0);

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                match ct.tensor(reader, "per_layer_token_embd.weight", device) {
                    Ok(ple_emb_qt) => {
                        let ple_emb = ple_emb_qt.dequantize(device)?;
                        let ple_proj = QMatMul::from_qtensor(ct.tensor(
                            reader,
                            "per_layer_model_proj.weight",
                            device,
                        )?)?;
                        let ple_norm = RmsNorm::from_qtensor(
                            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
                            rms_norm_eps,
                            device,
                        )?;
                        let total_ple_dim = block_count * ple_dim;
                        tracing::info!(
                            "PLE loaded: ple_dim={}, total_ple_dim={}, emb_shape={:?}",
                            ple_dim,
                            total_ple_dim,
                            ple_emb.shape(),
                        );
                        (
                            Some(Embedding::new(ple_emb, total_ple_dim)),
                            Some(ple_proj),
                            Some(ple_norm),
                        )
                    }
                    Err(_) => {
                        tracing::warn!(
                            "PLE dim={} but global PLE tensors not found in GGUF",
                            ple_dim
                        );
                        (None, None, None)
                    }
                }
            } else {
                (None, None, None)
            };

        let num_shared = layers
            .iter()
            .filter(|l| l.kv_shared_layer_index.is_some())
            .count();
        let num_sliding = layers.iter().filter(|l| l.is_sliding).count();
        let has_ple = embed_tokens_per_layer.is_some();
        let has_scalar = layers[0].layer_scalar.is_some();
        tracing::info!(
            "QuantizedTextModel loaded: {} layers ({} sliding, {} global, {} shared, first_shared={}), \
             global_head_dim={}, sliding_head_dim={}, vocab={}, kv_heads={}, PLE={}, scalar={}, softcap={:?}",
            block_count,
            num_sliding,
            block_count - num_sliding,
            num_shared,
            first_shared,
            global_head_dim,
            sliding_head_dim,
            tok_embeddings.dim(0)?,
            head_count_kv,
            has_ple,
            has_scalar,
            final_logit_softcapping,
        );

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            kv_caches,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            ple_dim,
            num_layers: block_count,
            final_logit_softcapping,
        })
    }

    /// Embed token IDs and scale by sqrt(hidden_size).
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.tok_embeddings.forward(input_ids)?;
        xs * (self.embedding_length as f64).sqrt()
    }

    /// Compute PLE (Per-Layer Embedding) inputs from token IDs and their embeddings.
    /// Returns `[B, S, num_layers, ple_dim]` if PLE is enabled, else `None`.
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

        let (b_sz, seq_len) = input_ids.dims2()?;
        let ple_embed_scale = (self.ple_dim as f64).sqrt();

        let token_identity = (emb_pl.forward(input_ids)? * ple_embed_scale)?;
        let shape = &[b_sz, seq_len, self.num_layers, self.ple_dim];
        let token_identity = token_identity.reshape(shape)?;

        let proj_scale = (self.embedding_length as f64).powf(-0.5);
        let context_proj = (proj.forward(inputs_embeds)? * proj_scale)?;
        let context_proj = context_proj.reshape(shape)?;
        let context_proj = norm.forward(&context_proj)?;

        let combined = ((context_proj + token_identity)? * std::f64::consts::FRAC_1_SQRT_2)?;
        Ok(Some(combined))
    }

    /// Forward pass from pre-computed embeddings (for multimodal injection).
    pub fn forward_embeds(
        &mut self,
        layer_in: &Tensor,
        index_pos: usize,
        per_layer_inputs: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = layer_in.dims3()?;
        let mut layer_in = layer_in.clone();

        if tracing::enabled!(tracing::Level::TRACE) && index_pos == 0 {
            let norm = layer_in
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()
                .unwrap_or(f32::NAN)
                .sqrt();
            tracing::trace!("embedding norm after scaling: {:.4}", norm);
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, layer_in.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(
                &x,
                attention_mask.as_ref(),
                index_pos,
                &mut self.kv_caches,
                i,
            )?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let mut x = (x + residual)?;

            // PLE injection (after attention + MLP)
            if let (Some(gate), Some(proj), Some(norm)) = (
                &layer.per_layer_input_gate,
                &layer.per_layer_projection,
                &layer.post_per_layer_input_norm,
            ) {
                if let Some(ref pli) = per_layer_inputs {
                    let residual = &x;
                    // Extract this layer's PLE slice: narrow on dim -2 (num_layers), then squeeze
                    let ndim = pli.dims().len();
                    let ple_slice = pli.narrow(ndim - 2, i, 1)?.squeeze(ndim - 2)?;
                    let gated = Activation::GeluPytorchTanh.forward(&gate.forward(&x)?)?;
                    let gated = (gated * ple_slice)?;
                    let projected = proj.forward(&gated)?;
                    let normed = norm.forward(&projected)?;
                    x = (residual + normed)?;
                }
            }

            // Per-layer scalar
            if let Some(scalar) = &layer.layer_scalar {
                x = x.broadcast_mul(scalar)?;
            }

            if tracing::enabled!(tracing::Level::TRACE) && index_pos == 0 {
                let norm = x
                    .to_dtype(DType::F32)?
                    .sqr()?
                    .mean_all()?
                    .to_scalar::<f32>()
                    .unwrap_or(f32::NAN)
                    .sqrt();
                if i < 2 || norm.is_nan() || norm.is_infinite() {
                    tracing::trace!(
                        "layer {} output norm: {:.4} ({})",
                        i,
                        norm,
                        if layer.is_sliding {
                            "sliding"
                        } else {
                            "global"
                        },
                    );
                }
                if norm.is_nan() {
                    tracing::error!(
                        "NaN detected at layer {} ({}) — aborting forward pass",
                        i,
                        if layer.is_sliding {
                            "sliding"
                        } else {
                            "global"
                        },
                    );
                }
            }

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?;

        if tracing::enabled!(tracing::Level::TRACE) && index_pos == 0 {
            let norm_val = x
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()
                .unwrap_or(f32::NAN)
                .sqrt();
            tracing::trace!("pre-norm hidden state norm: {:.4}", norm_val);
        }

        let x = self.norm.forward(&x)?;

        if tracing::enabled!(tracing::Level::TRACE) && index_pos == 0 {
            let norm_val = x
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()
                .unwrap_or(f32::NAN)
                .sqrt();
            tracing::trace!("post-norm hidden state norm: {:.4}", norm_val);

            // Check for NaN/Inf in the normed hidden state
            let flat: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let nan_count = flat.iter().filter(|v| v.is_nan()).count();
            let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                tracing::error!(
                    "post-norm: {} NaN, {} Inf out of {} values",
                    nan_count,
                    inf_count,
                    flat.len()
                );
            }
        }

        let logits = self.output.forward(&x)?;

        // Apply logit softcapping: sc * tanh(logits / sc)
        let logits = if let Some(sc) = self.final_logit_softcapping {
            let logits = logits.to_dtype(DType::F32)?;
            ((logits / sc)?.tanh()? * sc)?
        } else {
            logits
        };

        if tracing::enabled!(tracing::Level::TRACE) {
            let flat: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let nan_count = flat.iter().filter(|v| v.is_nan()).count();
            let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
            let mut indexed: Vec<(usize, f32)> = flat.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<_> = indexed
                .iter()
                .take(5)
                .map(|(i, v)| format!("{}:{:.2}", i, v))
                .collect();
            tracing::trace!(
                "logits: nan={}, inf={}, top5=[{}]",
                nan_count,
                inf_count,
                top5.join(", "),
            );
        }

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            *cache = None;
        }
    }

    pub fn kv_cache_seq_len(&self) -> usize {
        self.kv_caches
            .iter()
            .find_map(|c| c.as_ref().map(|(k, _)| k.dim(2).unwrap_or(0)))
            .unwrap_or(0)
    }

    pub fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>> {
        self.kv_caches
            .iter()
            .enumerate()
            .map(|(idx, cache)| {
                Ok(LayerKvSnapshot {
                    layer_idx: idx,
                    is_sliding: self.layers[idx].is_sliding,
                    k_data: cache.as_ref().map(|(k, _)| k.clone()),
                    v_data: cache.as_ref().map(|(_, v)| v.clone()),
                    offset: 0,
                    current_seq_len: cache
                        .as_ref()
                        .map(|(k, _)| k.dim(2).unwrap_or(0))
                        .unwrap_or(0),
                })
            })
            .collect()
    }

    pub fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()> {
        for snap in snapshots {
            if snap.layer_idx < self.kv_caches.len() {
                self.kv_caches[snap.layer_idx] = match (&snap.k_data, &snap.v_data) {
                    (Some(k), Some(v)) => Some((k.clone(), v.clone())),
                    _ => None,
                };
            }
        }
        Ok(())
    }
}

impl TextForward for QuantizedTextModel {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let embeds = self.embed_tokens(input_ids)?;
        let pli = self.compute_per_layer_inputs(input_ids, &embeds)?;
        self.forward_embeds(&embeds, seqlen_offset, pli.as_ref())
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }
}
