pub mod factory;
pub(crate) mod pipelines;

use local_inference_helpers::candle_core::{DType, Device, Tensor};

pub struct DescribeRequest {
    pub prompt: String,
    pub pixel_values: Tensor,
    pub image_grid_thw: Tensor,
    pub num_image_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

pub struct DescribeResponse {
    pub text: String,
    pub tokens_generated: usize,
}

pub trait InferenceEngine: Send {
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self, device: &Device, dtype: DType) -> anyhow::Result<()>;
    fn describe(&mut self, req: &DescribeRequest) -> anyhow::Result<DescribeResponse>;
}
