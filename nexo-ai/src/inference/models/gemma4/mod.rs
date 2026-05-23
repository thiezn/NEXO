#[cfg(feature = "candle")]
pub mod candle;
pub mod common;
#[cfg(feature = "candle")]
mod model;
#[cfg(feature = "mlx")]
pub mod openai;

#[cfg(feature = "candle")]
pub use candle::{gguf, safetensors};
pub use common::template;
#[cfg(feature = "candle")]
pub use common::{config, generation};

#[cfg(feature = "candle")]
pub use model::Gemma4Model;
