use crate::message::MediaSource;
use serde::{Deserialize, Serialize};

/// A request to generate one or more images from text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationPayload {
    /// The positive prompt used for generation.
    pub prompt: String,

    /// The negative prompt used to suppress unwanted attributes.
    pub negative_prompt: Option<String>,

    /// The generated image size.
    pub size: ImageGenerationSize,

    /// The number of images requested.
    pub sample_count: u32,

    /// The diffusion step count, if supported by the runtime.
    pub steps: Option<u32>,

    /// The guidance scale, if supported by the runtime.
    pub guidance_scale: Option<f32>,

    /// The deterministic random seed, if requested.
    pub seed: Option<u64>,
}

/// A transport-safe generated image payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GeneratedImage {
    /// The zero-based order of the image within the response.
    pub index: usize,

    /// The generated image content.
    pub source: MediaSource,

    /// The optional media type of the generated image.
    pub media_type: Option<String>,

    /// The generated image width, in pixels, if known.
    pub width: Option<u32>,

    /// The generated image height, in pixels, if known.
    pub height: Option<u32>,
}

/// The desired image size for an image generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationSize {
    /// The image width, in pixels.
    pub width: u32,

    /// The image height, in pixels.
    pub height: u32,
}
