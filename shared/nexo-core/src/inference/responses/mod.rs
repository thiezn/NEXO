//! Inference final response payload definitions for supported operations.

/// Embedding and token conversion final responses.
pub mod embed;
/// Image generation final responses.
pub mod image;
/// Multimodal generation final responses.
pub mod multimodal;
/// Speech generation final responses.
pub mod speech;

pub use embed::{DetokenizationResponse, EmbedResponse, EmbeddingVector, TokenizationResponse};
pub use image::{GeneratedImage, ImageGenerationResponse};
pub use multimodal::MultiModalResponse;
pub use speech::{GeneratedAudio, SpeechGenerationResponse};
