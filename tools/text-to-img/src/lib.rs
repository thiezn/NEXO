pub mod config;
pub mod generator;
pub mod inference;
pub mod manifest;
pub mod models;
pub(crate) mod prompt;

pub use generator::generate_images;
pub use models::{GeneratedImage, GenerationConfig, GenerationResult, OutputFormat};
