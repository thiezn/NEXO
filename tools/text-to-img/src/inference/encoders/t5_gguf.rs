//! Custom T5 encoder loader for GGUF files using the GGUF standard tensor naming convention.

use anyhow::Result;
use local_inference_helpers::candle_core::quantized::gguf_file;
use local_inference_helpers::candle_core::quantized::QTensor;
use local_inference_helpers::candle_core::{DType, Device, Module, Tensor, D};
use local_inference_helpers::candle_nn::Activation;
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

struct T5GatedFFN {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    act: Activation,
}

impl T5GatedFFN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden_gelu = self.act.forward(&self.gate.forward(xs)?)?;
        let hidden_linear = self.up.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        self.down.forward(&xs).map_err(Into::into)
    }
}

struct T5SelfAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<local_inference_helpers::candle_nn::Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
}

impl T5SelfAttention {
    fn forward(&mut self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self
            .q
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        let scores = q.matmul(&k.t()?)?;

        let (scores, position_bias) = match position_bias {
            Some(pb) => (scores.broadcast_add(pb)?, pb.clone()),
            None => match &self.relative_attention_bias {
                None => {
                    return Err(anyhow::anyhow!(
                        "no relative attention bias and none provided"
                    ));
                }
                Some(rel_attn_bias) => {
                    let pb = self.compute_bias(seq_len, rel_attn_bias, scores.device())?;
                    let pb = pb.to_dtype(scores.dtype())?;
                    (scores.broadcast_add(&pb)?, pb)
                }
            },
        };

        let attn_weights = local_inference_helpers::candle_nn::ops::softmax_last_dim(&scores)?;
        let v = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, self.n_heads * self.d_kv))?;
        let output = self.o.forward(&attn_output)?;
        Ok((output, position_bias))
    }

    fn compute_bias(
        &self,
        seq_len: usize,
        rel_attn_bias: &local_inference_helpers::candle_nn::Embedding,
        device: &Device,
    ) -> Result<Tensor> {
        let num_buckets = self.relative_attention_num_buckets as u32 / 2;
        let max_exact = num_buckets / 2;
        let relative_position: Vec<Vec<u32>> = (0..seq_len as u32)
            .map(|i| {
                (0..seq_len as u32)
                    .map(|j| {
                        if i < j {
                            if j - i < max_exact {
                                j - i + num_buckets
                            } else {
                                let b = f32::log(
                                    (j - i) as f32 / max_exact as f32,
                                    self.relative_attention_max_distance as f32 / max_exact as f32,
                                ) * (num_buckets - max_exact) as f32;
                                u32::min(
                                    max_exact + num_buckets + b as u32,
                                    self.relative_attention_num_buckets as u32 - 1,
                                )
                            }
                        } else if i - j < max_exact {
                            i - j
                        } else {
                            let b = f32::log(
                                (i - j) as f32 / max_exact as f32,
                                self.relative_attention_max_distance as f32 / max_exact as f32,
                            ) * (num_buckets - max_exact) as f32;
                            max_exact + b as u32
                        }
                    })
                    .collect()
            })
            .collect();
        let relative_buckets = Tensor::new(relative_position, device)?;
        let position_bias = rel_attn_bias
            .forward(&relative_buckets)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        Ok(position_bias)
    }
}

struct T5EncoderBlock {
    attn_norm: T5LayerNorm,
    self_attn: T5SelfAttention,
    ffn_norm: T5LayerNorm,
    ffn: T5GatedFFN,
}

impl T5EncoderBlock {
    fn forward(&mut self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let normed = self.attn_norm.forward(xs)?;
        let (attn_output, position_bias) = self.self_attn.forward(&normed, position_bias)?;
        let xs = (xs + attn_output)?;

        let normed = self.ffn_norm.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let xs = (xs + ffn_output)?;

        Ok((xs, position_bias))
    }
}

pub(crate) struct GgufT5Encoder {
    embedding: local_inference_helpers::candle_nn::Embedding,
    blocks: Vec<T5EncoderBlock>,
    final_norm: T5LayerNorm,
}

impl GgufT5Encoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
        for name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, name, device)?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }

        let get = |name: &str| -> Result<Arc<QTensor>> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
        };

        let emb_tensor = get("token_embd.weight")?;
        let emb_weights = emb_tensor.dequantize(device)?;
        let d_model = emb_weights.dim(1)?;
        let embedding = local_inference_helpers::candle_nn::Embedding::new(emb_weights, d_model);

        let n_layers = content
            .metadata
            .get("t5encoder.block_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(24);

        let num_heads = 64usize;
        let d_kv = 64usize;
        let eps = 1e-6f64;
        let rel_attn_buckets = 32usize;
        let rel_attn_max_dist = 128usize;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("enc.blk.{i}");

            let q = QMatMul::from_weights(get(&format!("{prefix}.attn_q.weight"))?)?;
            let k = QMatMul::from_weights(get(&format!("{prefix}.attn_k.weight"))?)?;
            let v = QMatMul::from_weights(get(&format!("{prefix}.attn_v.weight"))?)?;
            let o = QMatMul::from_weights(get(&format!("{prefix}.attn_o.weight"))?)?;

            let relative_attention_bias = if i == 0 {
                let rel_b = get(&format!("{prefix}.attn_rel_b.weight"))?;
                let rel_weights = rel_b.dequantize(device)?;
                let emb_dim = rel_weights.dim(1)?;
                Some(local_inference_helpers::candle_nn::Embedding::new(
                    rel_weights,
                    emb_dim,
                ))
            } else {
                None
            };

            let attn_norm_w = get(&format!("{prefix}.attn_norm.weight"))?.dequantize(device)?;
            let attn_norm = T5LayerNorm {
                weight: attn_norm_w,
                variance_epsilon: eps,
            };

            let self_attn = T5SelfAttention {
                q,
                k,
                v,
                o,
                n_heads: num_heads,
                d_kv,
                relative_attention_bias,
                relative_attention_num_buckets: rel_attn_buckets,
                relative_attention_max_distance: rel_attn_max_dist,
            };

            let gate = QMatMul::from_weights(get(&format!("{prefix}.ffn_gate.weight"))?)?;
            let up = QMatMul::from_weights(get(&format!("{prefix}.ffn_up.weight"))?)?;
            let down = QMatMul::from_weights(get(&format!("{prefix}.ffn_down.weight"))?)?;
            let ffn = T5GatedFFN {
                gate,
                up,
                down,
                act: Activation::NewGelu,
            };

            let ffn_norm_w = get(&format!("{prefix}.ffn_norm.weight"))?.dequantize(device)?;
            let ffn_norm = T5LayerNorm {
                weight: ffn_norm_w,
                variance_epsilon: eps,
            };

            blocks.push(T5EncoderBlock {
                attn_norm,
                self_attn,
                ffn_norm,
                ffn,
            });
        }

        let final_norm_w = get("enc.output_norm.weight")?.dequantize(device)?;
        let final_norm = T5LayerNorm {
            weight: final_norm_w,
            variance_epsilon: eps,
        };

        Ok(Self {
            embedding,
            blocks,
            final_norm,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let mut xs = self.embedding.forward(input_ids)?;
        let mut position_bias: Option<Tensor> = None;

        for block in &mut self.blocks {
            let (new_xs, new_pb) = block.forward(&xs, position_bias.as_ref())?;
            xs = new_xs;
            position_bias = Some(new_pb);
        }

        self.final_norm.forward(&xs)
    }
}
