use serde::{Deserialize, Serialize};

use crate::MediaSource;

/// The response returned for an image generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationResponse {
    /// The generated images.
    pub images: Vec<GeneratedImage>,
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
