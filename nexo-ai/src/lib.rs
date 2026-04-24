pub mod api;
pub mod audio;
pub mod config;
pub mod coordinator;
pub mod device;
pub mod download;
pub mod models;
#[cfg(feature = "mlx")]
pub mod openai;
pub mod registry;
#[cfg(feature = "mlx")]
pub mod servers;
pub mod statistics;
pub mod vision;

#[cfg(feature = "cli")]
pub mod cli;
