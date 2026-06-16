//! Inference request payload definitions for various modalities.

/// Embedding generation requests
pub mod embed;

/// Image generation and analysis requests
pub mod image;

/// Multimodal requests that combine text, image, and/or audio inputs and outputs.
pub mod multimodal;

/// Speech generation and analysis requests
pub mod speech;

pub use self::embed::{
    DetokenizationPayload, EmbedPayload, GenerationPromptPolicy, SpecialTokenPolicy,
    TokenizationPayload,
};
pub use self::image::{GeneratedImage, ImageGenerationPayload, ImageGenerationSize};
pub use self::speech::{AudioFormat, GeneratedAudio, SpeechGenerationPayload, SpeechLanguage};
pub use multimodal::MultiModalPayload;
