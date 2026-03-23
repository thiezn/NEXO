#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use local_inference_helpers::candle_core::{DType, Result, Tensor, D};
use local_inference_helpers::candle_nn::{linear_b, Linear, Module, VarBuilder};

use super::config::TextConfig;

/// Gated RMSNorm: norm(hidden) * SiLU(gate)
struct RMSNormGated {
    weight: Tensor,
    eps: f64,
}

impl RMSNormGated {
    fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let xs = hidden_states.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs_normed = xs_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        // SiLU gate
        let gate_f32 = gate.to_dtype(DType::F32)?;
        let silu_gate = (&gate_f32 * local_inference_helpers::candle_nn::ops::sigmoid(&gate_f32)?)?.to_dtype(dtype)?;
        xs_normed.mul(&silu_gate)
    }
}

/// L2-normalize along last dimension
fn l2norm(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-6)?;
    xs.broadcast_div(&norm)
}

/// Causal conv1d for prefill: apply depthwise conv1d with causal padding then SiLU
fn causal_conv1d_prefill(
    xs: &Tensor,       // (batch, channels, seq_len)
    weight: &Tensor,   // (channels, kernel_size)
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (_batch, _channels, seq_len) = xs.dims3()?;
    // Pad left with zeros for causal behavior
    let padded = xs.pad_with_zeros(2, kernel_size - 1, 0)?;
    // Depthwise conv1d: weight shape is (channels, 1, kernel_size)
    let weight_3d = weight.unsqueeze(1)?;
    let out = padded.conv1d(&weight_3d, 0, 1, 1, padded.dim(1)?)?;
    // Take only the last seq_len positions
    let out = out.narrow(2, 0, seq_len)?;
    // SiLU activation
    let silu = local_inference_helpers::candle_nn::Activation::Silu;
    let out = out.apply(&silu)?;
    // Conv state: last (kernel_size - 1) columns of padded input
    // Actually we want: the last (kernel_size-1) timesteps of the original input (padded to kernel_size-1)
    let conv_state = if seq_len >= kernel_size - 1 {
        xs.narrow(2, seq_len - (kernel_size - 1), kernel_size - 1)?
    } else {
        xs.pad_with_zeros(2, kernel_size - 1 - seq_len, 0)?
    };
    Ok((out, conv_state))
}

/// Causal conv1d update for single token decode
fn causal_conv1d_update(
    xs: &Tensor,         // (batch, channels, 1)
    conv_state: &Tensor, // (batch, channels, kernel_size - 1)
    weight: &Tensor,     // (channels, kernel_size)
) -> Result<(Tensor, Tensor)> {
    let (_batch, _channels, _) = xs.dims3()?;
    let state_len = conv_state.dim(2)?;
    // Concatenate state with new input
    let combined = Tensor::cat(&[conv_state, xs], 2)?;
    // New state: last state_len elements
    let new_state = combined.narrow(2, combined.dim(2)? - state_len, state_len)?;
    // Depthwise conv
    let weight_3d = weight.unsqueeze(1)?;
    let out = combined.conv1d(&weight_3d, 0, 1, 1, combined.dim(1)?)?;
    let out_len = out.dim(2)?;
    let out = out.narrow(2, out_len - 1, 1)?;
    // SiLU
    let silu = local_inference_helpers::candle_nn::Activation::Silu;
    let out = out.apply(&silu)?;
    Ok((out, new_state))
}

/// Recurrent gated delta rule - works for any sequence length
/// Used for single-token decode and as fallback for prefill
fn recurrent_gated_delta_rule(
    query: &Tensor,           // (batch, heads, seq_len, k_dim)
    key: &Tensor,             // (batch, heads, seq_len, k_dim)
    value: &Tensor,           // (batch, heads, seq_len, v_dim)
    g: &Tensor,               // (batch, heads, seq_len)
    beta: &Tensor,            // (batch, heads, seq_len)
    initial_state: Option<&Tensor>, // (batch, heads, k_dim, v_dim)
    output_final_state: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let query = l2norm(query)?;
    let key = l2norm(key)?;

    let (batch_size, num_heads, seq_len, k_dim) = query.dims4()?;
    let v_dim = value.dim(3)?;
    let scale = 1.0 / (k_dim as f64).sqrt();
    let query = (query * scale)?;

    let device = query.device();
    let mut state = match initial_state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((batch_size, num_heads, k_dim, v_dim), DType::F32, device)?,
    };

    let query = query.to_dtype(DType::F32)?;
    let key = key.to_dtype(DType::F32)?;
    let value = value.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;

    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        // Extract timestep slices: (batch, heads, dim)
        let q_t = query.narrow(2, i, 1)?.squeeze(2)?;
        let k_t = key.narrow(2, i, 1)?.squeeze(2)?;
        let v_t = value.narrow(2, i, 1)?.squeeze(2)?;
        let g_t = g.narrow(2, i, 1)?.squeeze(2)?; // (batch, heads)
        let beta_t = beta.narrow(2, i, 1)?.squeeze(2)?; // (batch, heads)

        // Decay: state = state * exp(g_t)
        let g_exp = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (batch, heads, 1, 1)
        state = state.broadcast_mul(&g_exp)?;

        // Retrieve: kv_mem = (state * k_t[:,:,:,None]).sum(dim=-2) → (batch, heads, v_dim)
        let k_expanded = k_t.unsqueeze(D::Minus1)?; // (batch, heads, k_dim, 1)
        let kv_mem = state.broadcast_mul(&k_expanded)?.sum(2)?; // (batch, heads, v_dim)

        // Delta update: delta = (v_t - kv_mem) * beta_t
        let beta_expanded = beta_t.unsqueeze(D::Minus1)?; // (batch, heads, 1)
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_expanded)?; // (batch, heads, v_dim)

        // State update: state += k_t[:,:,:,None] * delta[:,:,None,:]
        let delta_expanded = delta.unsqueeze(2)?; // (batch, heads, 1, v_dim)
        state = (state + k_expanded.broadcast_mul(&delta_expanded)?)?;

        // Output: out = (state * q_t[:,:,:,None]).sum(dim=-2) → (batch, heads, v_dim)
        let q_expanded = q_t.unsqueeze(D::Minus1)?; // (batch, heads, k_dim, 1)
        let out_t = state.broadcast_mul(&q_expanded)?.sum(2)?; // (batch, heads, v_dim)
        outputs.push(out_t.unsqueeze(2)?); // (batch, heads, 1, v_dim)
    }

    let out = Tensor::cat(&outputs, 2)?; // (batch, heads, seq_len, v_dim)
    let out = out
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(query.dtype())?;

    let final_state = if output_final_state {
        Some(state)
    } else {
        None
    };

    Ok((out, final_state))
}

pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: RMSNormGated,
    out_proj: Linear,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    // Cached state for decode
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNet {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_k_heads = cfg.linear_num_key_heads;
        let num_v_heads = cfg.linear_num_value_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;

        let in_proj_qkv = linear_b(cfg.hidden_size, conv_dim, false, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_b(cfg.hidden_size, value_dim, false, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_b(cfg.hidden_size, num_v_heads, false, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_b(cfg.hidden_size, num_v_heads, false, vb.pp("in_proj_a"))?;

        // Conv1d weight: stored as (conv_dim, 1, kernel_size) in PyTorch
        // We load it and squeeze to (conv_dim, kernel_size)
        let conv1d_weight = vb.pp("conv1d").get((conv_dim, 1, conv_kernel_size), "weight")?;
        let conv1d_weight = conv1d_weight.squeeze(1)?;

        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;

        let norm = RMSNormGated::new(head_v_dim, cfg.rms_norm_eps, vb.pp("norm"))?;
        let out_proj = linear_b(value_dim, cfg.hidden_size, false, vb.pp("out_proj"))?;

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

    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let is_decode = seq_len == 1 && self.conv_state.is_some();

        // Project QKV and auxiliary signals
        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?; // (b, s, conv_dim)
        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // (b, conv_dim, s)

        let z = self.in_proj_z.forward(hidden_states)?; // (b, s, value_dim)
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let b = self.in_proj_b.forward(hidden_states)?; // (b, s, num_v_heads)
        let a = self.in_proj_a.forward(hidden_states)?; // (b, s, num_v_heads)

        // Apply causal conv1d
        let mixed_qkv = if is_decode {
            let conv_state = self
                .conv_state
                .as_ref()
                .ok_or_else(|| local_inference_helpers::candle_core::Error::Msg("conv_state missing during decode".into()))?;
            let (out, new_state) =
                causal_conv1d_update(&mixed_qkv, conv_state, &self.conv1d_weight)?;
            self.conv_state = Some(new_state);
            out
        } else {
            let (out, conv_state) =
                causal_conv1d_prefill(&mixed_qkv, &self.conv1d_weight, self.conv_kernel_size)?;
            self.conv_state = Some(conv_state);
            out
        };

        // Transpose back and split into Q, K, V
        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // (b, s, conv_dim)
        let query = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let key = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let value = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let query = query.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let key = key.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let value = value.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // Compute gating: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
        let beta = local_inference_helpers::candle_nn::ops::sigmoid(&b)?;

        let a_f32 = a.to_dtype(DType::F32)?;
        let dt_bias = self.dt_bias.to_dtype(DType::F32)?;
        let a_log = self.a_log.to_dtype(DType::F32)?;
        let a_plus_bias = a_f32.broadcast_add(&dt_bias)?;
        let softplus = softplus_tensor(&a_plus_bias)?;
        let neg_a_exp = a_log.exp()?.neg()?;
        let g = softplus.broadcast_mul(&neg_a_exp)?;

        // Expand K heads to match V heads if needed
        let (query, key) = if self.num_v_heads > self.num_k_heads {
            let repeat = self.num_v_heads / self.num_k_heads;
            let query = repeat_interleave(&query, repeat, 2)?;
            let key = repeat_interleave(&key, repeat, 2)?;
            (query, key)
        } else {
            (query, key)
        };

        // Transpose to (batch, heads, seq_len, dim) for delta rule
        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?; // (batch, heads, seq_len)
        let g = g.transpose(1, 2)?;

        let (core_out, final_state) = recurrent_gated_delta_rule(
            &query,
            &key,
            &value,
            &g,
            &beta,
            self.recurrent_state.as_ref(),
            true, // always save state for potential decode
        )?;

        self.recurrent_state = final_state;

        // Reshape and apply gated norm — batch all heads together
        // core_out: (batch, seq, heads, v_dim) → (batch*seq*heads, v_dim)
        let core_out = core_out
            .reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z
            .reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?;
        let core_out = self.norm.forward(&core_out, &z)?;
        let core_out = core_out.reshape((batch_size, seq_len, self.value_dim))?;

        self.out_proj.forward(&core_out)
    }
}

/// softplus(x) = log(1 + exp(x))
fn softplus_tensor(xs: &Tensor) -> Result<Tensor> {
    let ones = xs.ones_like()?;
    let exp_x = xs.exp()?;
    (exp_x + ones)?.log()
}

/// Repeat interleave along a dimension
fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(xs.clone());
    }
    let dims = xs.dims();
    let n = dims[dim];
    let mut chunks = Vec::with_capacity(n * repeats);
    for i in 0..n {
        let slice = xs.narrow(dim, i, 1)?;
        for _ in 0..repeats {
            chunks.push(slice.clone());
        }
    }
    Tensor::cat(&chunks, dim)
}
