#[cfg(feature = "candle")]
pub mod flux2;
pub mod gemma4;
#[cfg(feature = "candle")]
pub mod qwen_image;
pub mod stub;
pub mod support;
#[cfg(feature = "mlx")]
pub mod voxtral;
#[cfg(any(feature = "candle", feature = "mlx"))]
pub mod whisper;
#[cfg(feature = "candle")]
pub mod z_image;
