#[cfg(feature = "candle")]
pub mod candle;
pub mod common;
mod model;
#[cfg(feature = "mlx")]
pub mod openai;

#[cfg(feature = "candle")]
pub use model::WhisperModel;
