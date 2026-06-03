use std::path::Path;

use super::ImageBuffer;

/// Load an image from a file path and return it as an RGB8 `ImageBuffer`.
///
/// The format is detected from the file extension. The image is always
/// converted to RGB8 regardless of the source format.
pub fn load_file(path: &Path) -> anyhow::Result<ImageBuffer> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open image at {}: {e}", path.display()))?
        .into_rgb8();

    let width = img.width();
    let height = img.height();
    let data = img.into_raw();

    Ok(ImageBuffer {
        data,
        width,
        height,
        channels: 3,
    })
}

/// Load an image from raw bytes (e.g. downloaded data) and return it as an RGB8 `ImageBuffer`.
///
/// The format is detected from the byte header (magic bytes).
pub fn load_bytes(data: &[u8]) -> anyhow::Result<ImageBuffer> {
    let img = image::load_from_memory(data)
        .map_err(|e| anyhow::anyhow!("failed to decode image from bytes: {e}"))?
        .into_rgb8();

    let width = img.width();
    let height = img.height();
    let raw = img.into_raw();

    Ok(ImageBuffer {
        data: raw,
        width,
        height,
        channels: 3,
    })
}
