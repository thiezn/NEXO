use std::io::Cursor;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{ColorType, ExtendedColorType, ImageEncoder};

use super::ImageBuffer;

/// Save an `ImageBuffer` to a file. The format is inferred from the file extension.
///
/// Supported extensions include `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, etc.
pub fn save_file(buffer: &ImageBuffer, path: &Path) -> anyhow::Result<()> {
    let color_type = match buffer.channels {
        3 => ColorType::Rgb8,
        4 => ColorType::Rgba8,
        c => anyhow::bail!("unsupported channel count for saving: {c}"),
    };

    image::save_buffer(path, &buffer.data, buffer.width, buffer.height, color_type)
        .map_err(|e| anyhow::anyhow!("failed to save image to {}: {e}", path.display()))?;

    Ok(())
}

/// Encode an `ImageBuffer` as PNG bytes in memory.
pub fn encode_png(buffer: &ImageBuffer) -> anyhow::Result<Vec<u8>> {
    let color_type = match buffer.channels {
        3 => ColorType::Rgb8,
        4 => ColorType::Rgba8,
        c => anyhow::bail!("unsupported channel count for PNG encoding: {c}"),
    };

    let mut out = Cursor::new(Vec::new());
    let encoder = PngEncoder::new(&mut out);
    encoder
        .write_image(
            &buffer.data,
            buffer.width,
            buffer.height,
            ExtendedColorType::from(color_type),
        )
        .map_err(|e| anyhow::anyhow!("PNG encoding failed: {e}"))?;

    Ok(out.into_inner())
}

/// Encode an `ImageBuffer` as JPEG bytes in memory with the given quality (1-100).
pub fn encode_jpeg(buffer: &ImageBuffer, quality: u8) -> anyhow::Result<Vec<u8>> {
    let color_type = match buffer.channels {
        3 => ColorType::Rgb8,
        4 => ColorType::Rgba8,
        c => anyhow::bail!("unsupported channel count for JPEG encoding: {c}"),
    };

    let mut out = Cursor::new(Vec::new());
    let encoder = JpegEncoder::new_with_quality(&mut out, quality);
    encoder
        .write_image(
            &buffer.data,
            buffer.width,
            buffer.height,
            ExtendedColorType::from(color_type),
        )
        .map_err(|e| anyhow::anyhow!("JPEG encoding failed: {e}"))?;

    Ok(out.into_inner())
}
