use serde::Serialize;

/// Output format for generated images.
#[derive(Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
}

impl OutputFormat {
    pub fn extension(&self) -> &str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    pub fn mime_type(&self) -> &str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

/// Configuration for the image generation pipeline.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub model: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub num_images: u32,
    pub output_format: OutputFormat,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model: "flux-schnell:q8".to_string(),
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: None,
            guidance: None,
            seed: None,
            num_images: 1,
            output_format: OutputFormat::default(),
        }
    }
}

/// Result of the image generation pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationResult {
    pub prompt_used: String,
    pub images: Vec<GeneratedImage>,
}

/// A single generated image as base64-encoded data.
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedImage {
    pub index: u32,
    pub base64_data: String,
    pub seed: u64,
}
