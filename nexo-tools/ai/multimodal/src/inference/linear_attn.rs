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
use mlx_rs::{Array, Dtype, array, transforms};

use crate::mlx_helpers::weight_loader::{get_weight, get_weight_as};
use crate::model_config::TextConfig;

// ---------------------------------------------------------------------------
// Gated RMSNorm: norm(hidden) * SiLU(gate)
// ---------------------------------------------------------------------------

struct RmsNormGated {
    weight: Array,
    eps: f64,
}

impl RmsNormGated {
    fn new(
        hidden_size: usize,
        eps: f64,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let weight = get_weight(weights, &format!("{prefix}.weight"))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, hidden_states: &Array, gate: &Array) -> Result<Array, Exception> {
        let dtype = hidden_states.dtype();
        let xs = hidden_states.as_dtype(Dtype::Float32)?;
        let variance = ops::mean(&ops::power(&xs, &array!(2.0f32))?, &[-1], true)?;
        let xs_normed = xs.multiply(ops::rsqrt(&variance.add(array!(self.eps as f32))?)?)?;
        let xs_normed = xs_normed.as_dtype(dtype)?.multiply(&self.weight)?;
        // SiLU gate
        let gate_f32 = gate.as_dtype(Dtype::Float32)?;
        let silu_gate = gate_f32.multiply(ops::sigmoid(&gate.as_dtype(Dtype::Float32)?)?)?.as_dtype(dtype)?;
        xs_normed.multiply(silu_gate)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn l2norm(xs: &Array) -> Result<Array, Exception> {
    let norm = ops::sqrt(&ops::sum(&ops::power(xs, &array!(2.0f32))?, &[-1], true)?)?;
    let norm = norm.add(array!(1e-6f32))?;
    xs.multiply(ops::reciprocal(&norm)?)
}

fn softplus(xs: &Array) -> Result<Array, Exception> {
    // softplus(x) = log(1 + exp(x))
    ops::log(&ops::exp(xs)?.add(array!(1.0f32))?)
}

// ---------------------------------------------------------------------------
// Causal Conv1d
// ---------------------------------------------------------------------------

/// Causal conv1d prefill: depthwise conv with left-padding then SiLU
fn causal_conv1d_prefill(
    xs: &Array,       // (batch, channels, seq_len)
    weight: &Array,   // (channels, 1, kernel_size)
    kernel_size: usize,
) -> Result<(Array, Array), Exception> {
    let shape = xs.shape();
    let seq_len = shape[2] as usize;
    let channels = shape[1];

    // Pad left with zeros for causal behavior
    let pad_size = (kernel_size - 1) as i32;
    let zeros = Array::zeros::<f32>(&[shape[0], channels, pad_size])?;
    let padded = concatenate(&[&zeros, xs], 2)?;

    // Depthwise conv1d: groups = channels
    let out = ops::conv1d(&padded, weight, 1, 0, 1, channels as i32)?;
    // Take only first seq_len positions
    let out = out.index((.., .., ..seq_len as i32));
    // SiLU activation
    let out = nn::silu(out)?;

    // Conv state: last (kernel_size-1) timesteps of original input
    let conv_state = if seq_len >= kernel_size - 1 {
        xs.index((.., .., (seq_len - (kernel_size - 1)) as i32..))
    } else {
        let pad = Array::zeros::<f32>(&[shape[0], channels, (kernel_size - 1 - seq_len) as i32])?;
        concatenate(&[&pad, xs], 2)?
    };

    Ok((out, conv_state))
}

/// Causal conv1d update for single-token decode
fn causal_conv1d_update(
    xs: &Array,         // (batch, channels, 1)
    conv_state: &Array, // (batch, channels, kernel_size - 1)
    weight: &Array,     // (channels, 1, kernel_size)
) -> Result<(Array, Array), Exception> {
    let channels = xs.shape()[1];
    let state_len = conv_state.shape()[2];

    // Concatenate state with new input
    let combined = concatenate(&[conv_state, xs], 2)?;
    // New state: last state_len elements
    let new_state = combined.index((.., .., (combined.shape()[2] - state_len)..));

    // Depthwise conv
    let out = ops::conv1d(&combined, weight, 1, 0, 1, channels as i32)?;
    let out_len = out.shape()[2];
    let out = out.index((.., .., (out_len - 1)..));
    // SiLU
    let out = nn::silu(out)?;

    Ok((out, new_state))
}

// ---------------------------------------------------------------------------
// Recurrent gated delta rule
// ---------------------------------------------------------------------------

/// CPU-side recurrent gated delta rule implementation.
/// Extracts data to CPU f32 vecs for the sequential recurrence.
fn recurrent_gated_delta_rule(
    query: &Array,              // (batch, heads, seq_len, k_dim)
    key: &Array,                // (batch, heads, seq_len, k_dim)
    value: &Array,              // (batch, heads, seq_len, v_dim)
    g: &Array,                  // (batch, heads, seq_len)
    beta: &Array,               // (batch, heads, seq_len)
    initial_state: Option<&Array>, // (batch, heads, k_dim, v_dim)
) -> Result<(Array, Array), Exception> {
    let query = l2norm(query)?;
    let key = l2norm(key)?;

    let q_shape = query.shape();
    let (batch_size, num_heads, seq_len, k_dim) =
        (q_shape[0] as usize, q_shape[1] as usize, q_shape[2] as usize, q_shape[3] as usize);
    let v_dim = value.shape()[3] as usize;
    let scale = 1.0 / (k_dim as f32).sqrt();
    let query = query.multiply(array!(scale))?;

    // Pull to CPU f32
    let query = query.as_dtype(Dtype::Float32)?;
    let key = key.as_dtype(Dtype::Float32)?;
    let value = value.as_dtype(Dtype::Float32)?;
    let g_arr = g.as_dtype(Dtype::Float32)?;
    let beta_arr = beta.as_dtype(Dtype::Float32)?;
    transforms::eval(&[&query, &key, &value, &g_arr, &beta_arr])?;

    let q: Vec<f32> = query.as_slice().to_vec();
    let k: Vec<f32> = key.as_slice().to_vec();
    let v: Vec<f32> = value.as_slice().to_vec();
    let g_vec: Vec<f32> = g_arr.as_slice().to_vec();
    let b_vec: Vec<f32> = beta_arr.as_slice().to_vec();

    let state_size = batch_size * num_heads * k_dim * v_dim;
    let mut state = if let Some(s) = initial_state {
        let s = s.as_dtype(Dtype::Float32)?;
        transforms::eval(&[&s])?;
        s.as_slice::<f32>().to_vec()
    } else {
        vec![0f32; state_size]
    };

    let out_size = batch_size * num_heads * seq_len * v_dim;
    let mut output = vec![0f32; out_size];

    let qkv_head_stride = seq_len * k_dim;
    let v_head_stride = seq_len * v_dim;
    let gb_head_stride = seq_len;
    let s_head_stride = k_dim * v_dim;

    for b in 0..batch_size {
        for h in 0..num_heads {
            let qk_base = (b * num_heads + h) * qkv_head_stride;
            let v_base = (b * num_heads + h) * v_head_stride;
            let gb_base = (b * num_heads + h) * gb_head_stride;
            let s_base = (b * num_heads + h) * s_head_stride;
            let o_base = (b * num_heads + h) * seq_len * v_dim;

            for t in 0..seq_len {
                let q_off = qk_base + t * k_dim;
                let k_off = qk_base + t * k_dim;
                let v_off = v_base + t * v_dim;
                let g_val = g_vec[gb_base + t];
                let beta_val = b_vec[gb_base + t];
                let decay = g_val.exp();

                // Decay state
                for idx in 0..s_head_stride {
                    state[s_base + idx] *= decay;
                }

                // Delta update: kv_mem[j] = sum_i(state[i,j] * k[i])
                // delta[j] = (v[j] - kv_mem[j]) * beta
                // state[i,j] += k[i] * delta[j]
                for j in 0..v_dim {
                    let mut kv_mem_j = 0f32;
                    for i in 0..k_dim {
                        kv_mem_j += state[s_base + i * v_dim + j] * k[k_off + i];
                    }
                    let delta_j = (v[v_off + j] - kv_mem_j) * beta_val;
                    for i in 0..k_dim {
                        state[s_base + i * v_dim + j] += k[k_off + i] * delta_j;
                    }
                }

                // Output: out[j] = sum_i(state[i,j] * q[i])
                let o_off = o_base + t * v_dim;
                for j in 0..v_dim {
                    let mut out_j = 0f32;
                    for i in 0..k_dim {
                        out_j += state[s_base + i * v_dim + j] * q[q_off + i];
                    }
                    output[o_off + j] = out_j;
                }
            }
        }
    }

    // Build output: (batch, heads, seq, v_dim) -> transpose to (batch, seq, heads, v_dim)
    let out_tensor = Array::from_slice(
        &output,
        &[batch_size as i32, num_heads as i32, seq_len as i32, v_dim as i32],
    )
    .transpose(&[0, 2, 1, 3])?;

    let final_state = Array::from_slice(
        &state,
        &[batch_size as i32, num_heads as i32, k_dim as i32, v_dim as i32],
    );

    Ok((out_tensor, final_state))
}

// ---------------------------------------------------------------------------
// Repeat interleave
// ---------------------------------------------------------------------------

fn repeat_interleave(xs: &Array, repeats: usize, dim: usize) -> Result<Array, Exception> {
    if repeats == 1 {
        return Ok(xs.clone());
    }
    let shape = xs.shape();
    let n = shape[dim as usize] as usize;
    let dim_i32 = dim as i32;
    let mut chunks = Vec::with_capacity(n * repeats);
    for i in 0..n {
        let slice = xs.index((.., .., i as i32..(i + 1) as i32, ..));
        for _ in 0..repeats {
            chunks.push(slice.clone());
        }
    }
    let refs: Vec<&Array> = chunks.iter().collect();
    concatenate(&refs, dim_i32)
}

// ---------------------------------------------------------------------------
// GatedDeltaNet
// ---------------------------------------------------------------------------

pub struct GatedDeltaNet {
    in_proj_qkv: nn::Linear,
    in_proj_z: nn::Linear,
    in_proj_b: nn::Linear,
    in_proj_a: nn::Linear,
    conv1d_weight: Array,
    dt_bias: Array,
    a_log: Array,
    norm: RmsNormGated,
    out_proj: nn::Linear,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_state: Option<Array>,
    recurrent_state: Option<Array>,
}

impl GatedDeltaNet {
    pub fn new(
        cfg: &TextConfig,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let num_k_heads = cfg.linear_num_key_heads;
        let num_v_heads = cfg.linear_num_value_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;
        let h = cfg.hidden_size as i32;

        let mut in_proj_qkv = nn::LinearBuilder::new(h, conv_dim as i32).bias(false).build()?;
        let mut in_proj_z = nn::LinearBuilder::new(h, value_dim as i32).bias(false).build()?;
        let mut in_proj_b = nn::LinearBuilder::new(h, num_v_heads as i32).bias(false).build()?;
        let mut in_proj_a = nn::LinearBuilder::new(h, num_v_heads as i32).bias(false).build()?;

        in_proj_qkv.weight = get_weight(weights, &format!("{prefix}.in_proj_qkv.weight"))?;
        in_proj_z.weight = get_weight(weights, &format!("{prefix}.in_proj_z.weight"))?;
        in_proj_b.weight = get_weight(weights, &format!("{prefix}.in_proj_b.weight"))?;
        in_proj_a.weight = get_weight(weights, &format!("{prefix}.in_proj_a.weight"))?;

        // Conv1d weight: stored as (conv_dim, 1, kernel_size), keep as-is for conv1d op
        let conv1d_weight = get_weight(weights, &format!("{prefix}.conv1d.weight"))?;

        let dt_bias = get_weight(weights, &format!("{prefix}.dt_bias"))?;
        let a_log = get_weight(weights, &format!("{prefix}.A_log"))?;

        let norm = RmsNormGated::new(head_v_dim, cfg.rms_norm_eps, weights, &format!("{prefix}.norm"))?;

        let mut out_proj = nn::LinearBuilder::new(value_dim as i32, h).bias(false).build()?;
        out_proj.weight = get_weight(weights, &format!("{prefix}.out_proj.weight"))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_state: None,
            recurrent_state: None,
        })
    }

    pub fn reset_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    pub fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len) = (shape[0], shape[1]);
        let is_decode = seq_len == 1 && self.conv_state.is_some();

        // Project QKV and auxiliary signals
        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?; // (b, s, conv_dim)
        let mixed_qkv = mixed_qkv.transpose(&[0, 2, 1])?; // (b, conv_dim, s)

        let z = self.in_proj_z.forward(hidden_states)?; // (b, s, value_dim)
        let z = z.reshape(&[batch_size, seq_len, self.num_v_heads as i32, self.head_v_dim as i32])?;

        let b = self.in_proj_b.forward(hidden_states)?; // (b, s, num_v_heads)
        let a = self.in_proj_a.forward(hidden_states)?; // (b, s, num_v_heads)

        // Apply causal conv1d
        let mixed_qkv = if is_decode {
            let conv_state = self.conv_state.as_ref().unwrap();
            let (out, new_state) = causal_conv1d_update(&mixed_qkv, conv_state, &self.conv1d_weight)?;
            self.conv_state = Some(new_state);
            out
        } else {
            let (out, conv_state) = causal_conv1d_prefill(&mixed_qkv, &self.conv1d_weight, self.conv_kernel_size)?;
            self.conv_state = Some(conv_state);
            out
        };

        // Transpose back and split into Q, K, V
        let mixed_qkv = mixed_qkv.transpose(&[0, 2, 1])?; // (b, s, conv_dim)
        let kd = self.key_dim as i32;
        let vd = self.value_dim as i32;
        let query = mixed_qkv.index((.., .., ..kd));
        let key = mixed_qkv.index((.., .., kd..kd * 2));
        let value = mixed_qkv.index((.., .., kd * 2..));

        let nk = self.num_k_heads as i32;
        let nv = self.num_v_heads as i32;
        let hkd = self.head_k_dim as i32;
        let hvd = self.head_v_dim as i32;

        let query = query.reshape(&[batch_size, seq_len, nk, hkd])?;
        let key = key.reshape(&[batch_size, seq_len, nk, hkd])?;
        let value = value.reshape(&[batch_size, seq_len, nv, hvd])?;

        // Gating: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
        let beta = ops::sigmoid(&b)?;

        let a_f32 = a.as_dtype(Dtype::Float32)?;
        let dt_bias = self.dt_bias.as_dtype(Dtype::Float32)?;
        let a_log = self.a_log.as_dtype(Dtype::Float32)?;
        let a_plus_bias = a_f32.add(dt_bias)?;
        let sp = softplus(&a_plus_bias)?;
        let neg_a_exp = ops::negative(&ops::exp(&a_log)?)?;
        let g = sp.multiply(neg_a_exp)?;

        // Expand K heads to match V heads if needed
        let (query, key) = if self.num_v_heads > self.num_k_heads {
            let repeat = self.num_v_heads / self.num_k_heads;
            let query = repeat_interleave(&query, repeat, 2)?;
            let key = repeat_interleave(&key, repeat, 2)?;
            (query, key)
        } else {
            (query, key)
        };

        // Transpose to (batch, heads, seq_len, dim)
        let query = query.transpose(&[0, 2, 1, 3])?;
        let key = key.transpose(&[0, 2, 1, 3])?;
        let value = value.transpose(&[0, 2, 1, 3])?;
        let beta = beta.transpose(&[0, 2, 1])?; // (batch, heads, seq_len)
        let g = g.transpose(&[0, 2, 1])?;

        let (core_out, final_state) = recurrent_gated_delta_rule(
            &query,
            &key,
            &value,
            &g,
            &beta,
            self.recurrent_state.as_ref(),
        )?;

        self.recurrent_state = Some(final_state);

        // Reshape and apply gated norm
        let core_out = core_out.reshape(&[
            batch_size * seq_len * nv,
            hvd,
        ])?;
        let z = z.reshape(&[batch_size * seq_len * nv, hvd])?;
        let core_out = self.norm.forward(&core_out, &z)?;
        let core_out = core_out.reshape(&[batch_size, seq_len, vd])?;

        self.out_proj.forward(&core_out)
    }
}
