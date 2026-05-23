pub mod api;
pub mod audio;
pub mod config;
pub mod coordinator;
pub mod download;
pub mod inference;
pub mod registry;
pub mod statistics;
pub mod vision;

#[cfg(feature = "cli")]
pub mod cli;
