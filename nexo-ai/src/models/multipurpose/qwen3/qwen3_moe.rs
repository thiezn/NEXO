//! Vendored Qwen3 MoE quantized model.
//!
//! Based on `candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE`.
//! Vendored to add `clear_kv_cache()` which the upstream doesn't expose.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Activation, Embedding, Linear, Module};
use candle_transformers::fused_moe::FusedMoeGGUF;
use candle_transformers::models::quantized_qwen3::{Gguf, RotaryEmbedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::io::{Read, Seek};
use std::sync::Arc;

struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoeGGUF),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
        }
    }
}

struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    bq: Option<Tensor>,
    bk: Option<Tensor>,
    bv: Option<Tensor>,
    wo: QMatMul,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    num_kv_groups: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    dtype: DType,
    kv_cache: ConcatKvCache,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Seek + Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        device: &Device,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let wq = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let wk = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let wv = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;

        let bq = gg
            .tensor(&format!("{prefix}.attn_q.bias"))
            .ok()
            .map(|t| t.dequantize(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;
        let bk = gg
            .tensor(&format!("{prefix}.attn_k.bias"))
            .ok()
            .map(|t| t.dequantize(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;
        let bv = gg
            .tensor(&format!("{prefix}.attn_v.bias"))
            .ok()
            .map(|t| t.dequantize(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let wo = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = Some(gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?);
        let k_norm = Some(gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?);
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            wq,
            wk,
            wv,
            bq,
            bk,
            bv,
            wo,
            q_norm,
            k_norm,
            n_head: num_heads,
            n_kv_head: num_kv_heads,
            head_dim,
            num_kv_groups,
            rotary_emb,
            dtype,
            kv_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q = match &self.bq {
            Some(bq) => q.broadcast_add(bq)?,
            None => q,
        };
        let k = match &self.bk {
            Some(bk) => k.broadcast_add(bk)?,
            None => k,
        };
        let v = match &self.bv {
            Some(bv) => v.broadcast_add(bv)?,
            None => v,
        };

        let q = q
            .reshape((1, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            let q_flat = q.flatten(0, 2)?;
            let k_flat = k.flatten(0, 2)?;
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;
            let q = q_flat.reshape((1, self.n_head, seq_len, self.head_dim))?;
            let k = k_flat.reshape((1, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            let mask = if m.dtype() != scores.dtype() {
                m.to_dtype(scores.dtype())?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let out = ctx
            .transpose(1, 2)?
            .reshape((b, seq_len, self.n_head * self.head_dim))?;

        self.wo.forward(&out.to_dtype(in_dtype)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    fn kv_cache_seq_len(&self) -> usize {
        self.kv_cache.current_seq_len()
    }

    fn truncate_kv_cache(&mut self, len: usize) {
        let dim = self.kv_cache.dim();
        if let Some(k) = self.kv_cache.k_mut() {
            if let Ok(narrowed) = k.narrow(dim, 0, len) {
                *k = narrowed;
            }
        }
        if let Some(v) = self.kv_cache.v_mut() {
            if let Ok(narrowed) = v.narrow(dim, 0, len) {
                *v = narrowed;
            }
        }
    }
}

struct LayerWeights {
    self_attn: Attention,
    attention_norm: RmsNorm,
    mlp: MoeOrMlp,
    ffn_norm: RmsNorm,
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    dtype: DType,
    device: Device,
    cached_mask: Option<(usize, usize, usize, Tensor)>,
}

impl ModelWeights {
    pub fn from_gguf<R: Seek + Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        use candle_transformers::fused_moe::MoeCfg;

        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let head_count_kv =
            md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let embedding_length =
            md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let head_dim = md_get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let context_length =
            md_get(&format!("{arch}.context_length"))?.to_u32()? as usize;
        let block_count = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let rms_norm_eps = md_get(&format!("{arch}.attention.layer_norm_rms_epsilon"))?
            .to_f32()? as f64;
        let rope_freq_base = md_get(&format!("{arch}.rope.freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let shared_expert_intermediate_size =
            md_get(&format!("{arch}.expert_shared_feed_forward_length"))
                .and_then(|v| v.to_u32())
                .ok()
                .and_then(|v| if v > 0 { Some(v as usize) } else { None });

        let moe_intermediate_size =
            md_get(&format!("{arch}.expert_feed_forward_length"))?.to_u32()? as usize;
        let num_experts = md_get(&format!("{arch}.expert_count"))?.to_u32()? as usize;
        let num_experts_per_tok =
            md_get(&format!("{arch}.expert_used_count"))?.to_u32()? as usize;

        let moe_cfg = MoeCfg {
            moe_intermediate_size,
            num_experts,
            norm_topk_prob: shared_expert_intermediate_size.is_none(),
            num_experts_per_tok,
            hidden_size: head_dim,
            act: Activation::Silu,
            decoder_sparse_step: None,
        };

        let tok_embeddings = gg.tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let output = match gg.qmatmul("output.weight") {
            Ok(v) => v,
            _ => gg.qmatmul("token_embd.weight")?,
        };

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            context_length,
            rope_freq_base as f64,
            device,
        )?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let mlp = if moe_cfg.num_experts > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0
            {
                let gate_ws = gg
                    .tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let gate = Linear::new(gate_ws, None);
                let gate_experts =
                    Arc::new(gg.tensor(&format!("{prefix}.ffn_gate_exps.weight"))?);
                let up_experts =
                    Arc::new(gg.tensor(&format!("{prefix}.ffn_up_exps.weight"))?);
                let down_experts =
                    Arc::new(gg.tensor(&format!("{prefix}.ffn_down_exps.weight"))?);
                MoeOrMlp::FusedMoe(FusedMoeGGUF {
                    gate,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    dtype,
                })
            } else {
                let feed_forward_w1 = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
                let feed_forward_w2 = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
                let feed_forward_w3 = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
                MoeOrMlp::Mlp(Mlp {
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                })
            };

            let attention_norm =
                gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ffn_norm = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
            let self_attn = Attention::new(
                &mut gg,
                &prefix,
                dtype,
                head_count,
                head_count_kv,
                head_dim,
                rms_norm_eps,
                device,
                rotary_emb.clone(),
            )?;
            layers.push(LayerWeights {
                self_attn,
                attention_norm,
                mlp,
                ffn_norm,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            dtype,
            device: device.clone(),
            cached_mask: None,
        })
    }

    fn causal_mask(&mut self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        if let Some((cb, ct, co, ref tensor)) = self.cached_mask {
            if cb == b && ct == tgt && co == offset {
                return Ok(tensor.clone());
            }
        }
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    if j <= i + offset { 0. } else { minf }
                })
            })
            .collect();
        let tensor = Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?
            .to_dtype(self.dtype)?;
        self.cached_mask = Some((b, tgt, offset, tensor));
        Ok(self.cached_mask.as_ref().unwrap().3.clone())
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let mut xs = self.tok_embeddings.forward(x)?;
        let (b, l) = x.dims2()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            let residual = &xs;
            let x = layer.attention_norm.forward(&xs)?;
            let attn = layer.self_attn.forward(&x, causal_mask.as_ref(), offset)?;
            let x = (attn + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x, causal_mask.is_some())?;
            xs = (x + residual)?;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }

    pub fn embed_tokens(&self, input: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input)
    }

    pub fn forward_embeds(&mut self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let mut h = xs.clone();
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            let residual = &h;
            let x = layer.attention_norm.forward(&h)?;
            let attn = layer.self_attn.forward(&x, causal_mask.as_ref(), offset)?;
            let x = (attn + residual)?;
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x, causal_mask.is_some())?;
            h = (x + residual)?;
        }
        let h = h.narrow(1, l - 1, 1)?;
        let h = self.norm.forward(&h)?;
        self.output.forward(&h)?.to_dtype(DType::F32)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.clear_kv_cache();
        }
    }
}

impl crate::shared::templates::kv_cache::KvCacheState for ModelWeights {
    fn cache_token_count(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.kv_cache_seq_len())
            .unwrap_or(0)
    }

    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn truncate_to(&mut self, len: usize) {
        for layer in &mut self.layers {
            layer.self_attn.truncate_kv_cache(len);
        }
    }
}
