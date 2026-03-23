#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use local_inference_helpers::candle_core::{Result, Tensor, D};
use local_inference_helpers::candle_nn::{linear_b, Activation, Linear, Module, VarBuilder};

use super::config::TextConfig;

/// Collection of expert weights stored as 3D tensors
struct Experts {
    gate_up_proj: Tensor, // (num_experts, 2 * intermediate, hidden)
    down_proj: Tensor,    // (num_experts, hidden, intermediate)
    num_experts: usize,
    act_fn: Activation,
}

impl Experts {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(1);
        let moe_intermediate = cfg.moe_intermediate_size.unwrap_or(0);
        let gate_up_proj =
            vb.get((num_experts, 2 * moe_intermediate, cfg.hidden_size), "gate_up_proj")?;
        let down_proj =
            vb.get((num_experts, cfg.hidden_size, moe_intermediate), "down_proj")?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            num_experts,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor, // (tokens, hidden)
        top_k_index: &Tensor,   // (tokens, top_k) u32
        top_k_weights: &Tensor, // (tokens, top_k) f32
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let (num_tokens, hidden_dim) = hidden_states.dims2()?;
        let mut final_out = Tensor::zeros((num_tokens, hidden_dim), dtype, device)?;

        let top_k_index_vec = top_k_index.to_vec2::<u32>()?;
        let top_k_weights_vec = top_k_weights.to_vec2::<f32>()?;
        // Group tokens by expert
        let mut expert_tokens: Vec<Vec<(usize, usize)>> = vec![vec![]; self.num_experts];
        for (token_idx, (indices, _weights)) in top_k_index_vec
            .iter()
            .zip(top_k_weights_vec.iter())
            .enumerate()
        {
            for (k_pos, &expert_idx) in indices.iter().enumerate() {
                if (expert_idx as usize) < self.num_experts {
                    expert_tokens[expert_idx as usize].push((token_idx, k_pos));
                }
            }
        }

        // Process each expert
        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let token_indices: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
            let weights: Vec<f32> = tokens
                .iter()
                .map(|(t, k)| top_k_weights_vec[*t][*k])
                .collect();

            let token_idx_tensor =
                Tensor::from_vec(token_indices, (tokens.len(),), device)?;
            let weights_tensor =
                Tensor::from_vec(weights, (tokens.len(), 1), device)?.to_dtype(dtype)?;

            let current_state =
                hidden_states.index_select(&token_idx_tensor, 0)?;

            // Get expert weights
            let gate_up_w = self.gate_up_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;
            let down_w = self.down_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;

            // Forward through expert
            let gate_up = current_state.matmul(&gate_up_w.t()?)?;
            let intermediate = gate_up.dim(D::Minus1)? / 2;
            let gate = gate_up.narrow(D::Minus1, 0, intermediate)?;
            let up = gate_up.narrow(D::Minus1, intermediate, intermediate)?;
            let expert_out = gate
                .apply(&self.act_fn)?
                .mul(&up)?
                .matmul(&down_w.t()?)?;

            let weighted = expert_out.broadcast_mul(&weights_tensor)?;
            final_out = final_out.index_add(&token_idx_tensor, &weighted, 0)?;
        }

        Ok(final_out)
    }
}

/// Top-K router
struct TopKRouter {
    weight: Tensor, // (num_experts, hidden_size)
    top_k: usize,
}

impl TopKRouter {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(1);
        let weight = vb.get((num_experts, cfg.hidden_size), "weight")?;
        Ok(Self {
            weight,
            top_k: cfg.num_experts_per_tok.unwrap_or(1),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        // hidden_states: (tokens, hidden)
        let router_logits = hidden_states.matmul(&self.weight.t()?)?; // (tokens, num_experts)
        let routing_weights = local_inference_helpers::candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Top-k selection
        let top_k_indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;
        let top_k_values = routing_weights.gather(&top_k_indices, D::Minus1)?;

        // Normalize top-k weights
        let sum = top_k_values.sum_keepdim(D::Minus1)?;
        let top_k_values = top_k_values.broadcast_div(&sum)?;

        Ok((top_k_values, top_k_indices))
    }
}

/// Shared expert MLP
struct SharedExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl SharedExpert {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let shared_intermediate = cfg.shared_expert_intermediate_size.unwrap_or(0);
        let gate_proj = linear_b(cfg.hidden_size, shared_intermediate, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(cfg.hidden_size, shared_intermediate, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(shared_intermediate, cfg.hidden_size, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

/// Sparse MoE Block: router + experts + shared expert with gating
pub struct MoeMlp {
    gate: TopKRouter,
    experts: Experts,
    shared_expert: SharedExpert,
    shared_expert_gate: Linear,
}

impl MoeMlp {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate = TopKRouter::new(cfg, vb.pp("gate"))?;
        let experts = Experts::new(cfg, vb.pp("experts"))?;
        let shared_expert = SharedExpert::new(cfg, vb.pp("shared_expert"))?;
        let shared_expert_gate =
            linear_b(cfg.hidden_size, 1, false, vb.pp("shared_expert_gate"))?;

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let flat = hidden_states.reshape((batch_size * seq_len, hidden_dim))?;

        // Shared expert
        let shared_out = self.shared_expert.forward(&flat)?;
        let shared_gate = local_inference_helpers::candle_nn::ops::sigmoid(
            &self.shared_expert_gate.forward(&flat)?,
        )?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        // Routed experts
        let (top_k_weights, top_k_indices) = self.gate.forward(&flat)?;
        let expert_out = self.experts.forward(&flat, &top_k_indices, &top_k_weights)?;

        let out = (expert_out + shared_out)?;
        out.reshape((batch_size, seq_len, hidden_dim))
    }
}
