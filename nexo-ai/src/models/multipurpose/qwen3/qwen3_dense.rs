//! Vendored Qwen3 dense quantized model.
//!
//! Based on `candle_transformers::models::quantized_qwen3::ModelWeights`.
//! Vendored to add `embed_tokens()` and `forward_embeds()` for vision support.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::models::quantized_qwen3::{Gguf, RotaryEmbedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::io::{Read, Seek};
use std::sync::Arc;

struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Seek + Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;
        let kv_cache = ConcatKvCache::new(2);

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
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

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
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
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
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    cached_mask: Option<(usize, usize, usize, Tensor)>,
}

impl ModelWeights {
    pub fn from_gguf<R: Seek + Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let arch = md_get("general.architecture")?.to_string()?;
        let num_heads = md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads =
            md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let head_dim = md_get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(
                md_get(&format!("{arch}.embedding_length"))
                    .and_then(|v| v.to_u32())
                    .unwrap_or(128) as usize
                    / num_heads,
            );
        let num_layers = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let context_length = md_get(&format!("{arch}.context_length"))?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq_base = md_get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32())
            .unwrap_or(10000f32) as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let tok_embeddings = gg.tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let embedding_length = tok_embeddings.dim(1)?;

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            context_length,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");
            let self_attn = Attention::new(
                &mut gg,
                &prefix,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
            )?;
            let mlp = Mlp {
                gate_proj: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
                up_proj: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
                down_proj: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            };
            let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
            layers.push(LayerWeights {
                self_attn,
                mlp,
                ln1,
                ln2,
            });
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head = match gg.qmatmul("output.weight") {
            Ok(v) => v,
            _ => gg.qmatmul("token_embd.weight")?,
        };

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
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
        // SAFETY: we just assigned to cached_mask above, so unwrap is fine.
        Ok(self.cached_mask.as_ref().unwrap().3.clone())
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = x.dims2()?;
        let mut h = self.tok_embeddings.forward(x)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            let residual = &h;
            let x = layer.ln1.forward(&h)?;
            let x = layer.self_attn.forward(&x, causal_mask.as_ref(), offset)?;
            let x = (x + residual)?;
            let residual = &x;
            let ff = layer.ln2.forward(&x)?;
            let ff = ff.apply(&layer.mlp)?;
            h = (ff + residual)?;
        }
        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last)?.to_dtype(DType::F32)?.squeeze(1)
    }

    /// Forward pass returning the normalized hidden states for all positions.
    /// Used by embedding models that need the hidden representation, not logits.
    pub fn forward_hidden(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = x.dims2()?;
        let mut h = self.tok_embeddings.forward(x)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            let residual = &h;
            let x = layer.ln1.forward(&h)?;
            let x = layer.self_attn.forward(&x, causal_mask.as_ref(), offset)?;
            let x = (x + residual)?;
            let residual = &x;
            let ff = layer.ln2.forward(&x)?;
            let ff = ff.apply(&layer.mlp)?;
            h = (ff + residual)?;
        }
        self.norm.forward(&h)?.to_dtype(DType::F32)
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
            let x = layer.ln1.forward(&h)?;
            let x = layer.self_attn.forward(&x, causal_mask.as_ref(), offset)?;
            let x = (x + residual)?;
            let residual = &x;
            let ff = layer.ln2.forward(&x)?;
            let ff = ff.apply(&layer.mlp)?;
            h = (ff + residual)?;
        }
        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last)?.to_dtype(DType::F32)?.squeeze(1)
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
