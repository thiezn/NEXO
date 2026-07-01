use image::DynamicImage;

use super::ImageBuffer;

/// Configuration for terminal image display.
pub struct DisplayConfig {
    /// Maximum display width in terminal columns. `None` for auto.
    pub max_width: Option<u32>,
    /// Maximum display height in terminal rows. `None` for auto.
    pub max_height: Option<u32>,
    /// Whether to use truecolor (24-bit) output.
    pub truecolor: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            max_width: None,
            max_height: None,
            truecolor: true,
        }
    }
}

/// Display an `ImageBuffer` in the terminal using viuer.
pub fn display_in_terminal(buffer: &ImageBuffer, config: &DisplayConfig) -> anyhow::Result<()> {
    let dynamic_image = match buffer.channels {
        3 => {
            let img =
                image::RgbImage::from_raw(buffer.width, buffer.height, buffer.data.clone())
                    .ok_or_else(|| anyhow::anyhow!("failed to create RgbImage from buffer data"))?;
            DynamicImage::ImageRgb8(img)
        }
        4 => {
            let img = image::RgbaImage::from_raw(buffer.width, buffer.height, buffer.data.clone())
                .ok_or_else(|| anyhow::anyhow!("failed to create RgbaImage from buffer data"))?;
            DynamicImage::ImageRgba8(img)
        }
        c => anyhow::bail!("unsupported channel count for display: {c}"),
    };

    let viuer_config = viuer::Config {
        width: config.max_width,
        height: config.max_height,
        truecolor: config.truecolor,
        absolute_offset: false,
        ..Default::default()
    };

    viuer::print(&dynamic_image, &viuer_config)
        .map_err(|e| anyhow::anyhow!("failed to display image in terminal: {e}"))?;

    Ok(())
}
