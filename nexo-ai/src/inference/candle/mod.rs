//! Generic Candle-specific helpers reused across model families.

pub mod device;
#[cfg(feature = "candle")]
pub mod kv_cache;
#[cfg(feature = "candle")]
pub mod weights;
