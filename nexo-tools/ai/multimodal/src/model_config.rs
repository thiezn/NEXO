#![allow(dead_code)]

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
    Gelu,
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

fn default_depth() -> usize { 27 }
fn default_vision_hidden_size() -> usize { 1152 }
fn default_out_hidden_size() -> usize { 4096 }
fn default_vision_hidden_act() -> Activation { Activation::GeluPytorchTanh }
fn default_intermediate_size() -> usize { 4304 }
fn default_num_heads() -> usize { 16 }
fn default_in_channels() -> usize { 3 }
fn default_patch_size() -> usize { 16 }
fn default_spatial_merge_size() -> usize { 2 }
fn default_temporal_patch_size() -> usize { 2 }
fn default_num_position_embeddings() -> usize { 2304 }
fn default_deepstack_visual_indexes() -> Vec<usize> { Vec::new() }
fn default_hidden_act() -> Activation { Activation::Silu }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 10_000_000.0 }
fn default_max_position_embeddings() -> usize { 262144 }
fn default_full_attention_interval() -> usize { 4 }
fn default_linear_conv_kernel_dim() -> usize { 4 }
fn default_partial_rotary_factor() -> f64 { 0.25 }
fn default_mrope_section() -> Vec<usize> { vec![11, 11, 10] }

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(alias = "in_chans", default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_deepstack_visual_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            mrope_interleaved: true,
            mrope_section: default_mrope_section(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
}

impl TextConfig {
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some_and(|n| n > 1)
    }

    pub fn mlp_intermediate_size(&self) -> usize {
        self.intermediate_size
            .or(self.moe_intermediate_size)
            .unwrap_or(0)
    }

    pub fn is_linear_attention_layer(&self, layer_idx: usize) -> bool {
        if !self.layer_types.is_empty() {
            self.layer_types
                .get(layer_idx)
                .is_some_and(|t| t == "linear_attention")
        } else {
            layer_idx % self.full_attention_interval != (self.full_attention_interval - 1)
        }
    }

    pub fn rope_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}
