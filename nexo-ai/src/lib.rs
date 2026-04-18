pub mod audio;
pub mod config;
pub mod coordinator;
pub mod device;
pub mod download;
#[cfg(feature = "candle")]
pub mod models;
pub mod registry;
#[cfg(feature = "mlx")]
pub mod remote_models;
pub mod shared;
pub mod statistics;
pub mod vision;

#[cfg(feature = "cli")]
pub mod cli;
