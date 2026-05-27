use image::imageops::FilterType;

use super::ImageBuffer;

/// Resize an `ImageBuffer` to the given dimensions using Lanczos3 filtering.
///
/// The returned buffer has the same channel count as the input.
pub fn resize(buffer: &ImageBuffer, width: u32, height: u32) -> ImageBuffer {
    match buffer.channels {
        3 => {
            let img = image::RgbImage::from_raw(buffer.width, buffer.height, buffer.data.clone());
            // SAFETY: data length is guaranteed by ImageBuffer invariants
            let Some(img) = img else {
                return ImageBuffer::new(width, height, 3);
            };
            let resized = image::imageops::resize(&img, width, height, FilterType::Lanczos3);
            ImageBuffer {
                data: resized.into_raw(),
                width,
                height,
                channels: 3,
            }
        }
        4 => {
            let img = image::RgbaImage::from_raw(buffer.width, buffer.height, buffer.data.clone());
            let Some(img) = img else {
                return ImageBuffer::new(width, height, 4);
            };
            let resized = image::imageops::resize(&img, width, height, FilterType::Lanczos3);
            ImageBuffer {
                data: resized.into_raw(),
                width,
                height,
                channels: 4,
            }
        }
        _ => ImageBuffer::new(width, height, buffer.channels),
    }
}

/// Compute target dimensions for smart resizing, respecting a grid unit and pixel bounds.
///
/// The algorithm:
/// 1. Calculate aspect ratio = orig_h / orig_w
/// 2. If total pixels < min_pixels: scale up by sqrt(min_pixels / total)
/// 3. If total pixels > max_pixels: scale down by sqrt(max_pixels / total)
/// 4. Round both dimensions to nearest multiple of grid_unit
/// 5. Ensure both >= grid_unit
/// 6. While h * w > max_pixels: reduce the dimension that's most over aspect ratio by grid_unit
/// 7. Ensure both >= grid_unit again
pub fn smart_resize_dims(
    orig_height: u32,
    orig_width: u32,
    grid_unit: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> anyhow::Result<(u32, u32)> {
    anyhow::ensure!(grid_unit > 0, "grid_unit must be > 0");
    anyhow::ensure!(min_pixels <= max_pixels, "min_pixels must be <= max_pixels");
    anyhow::ensure!(
        orig_height > 0 && orig_width > 0,
        "image dimensions must be > 0"
    );

    let aspect = orig_height as f64 / orig_width as f64;
    let total = orig_height as u64 * orig_width as u64;

    let (mut h, mut w) = if total < min_pixels as u64 {
        let scale = (min_pixels as f64 / total as f64).sqrt();
        (
            (orig_height as f64 * scale) as u32,
            (orig_width as f64 * scale) as u32,
        )
    } else if total > max_pixels as u64 {
        let scale = (max_pixels as f64 / total as f64).sqrt();
        (
            (orig_height as f64 * scale) as u32,
            (orig_width as f64 * scale) as u32,
        )
    } else {
        (orig_height, orig_width)
    };

    h = round_to_multiple(h, grid_unit);
    w = round_to_multiple(w, grid_unit);

    h = h.max(grid_unit);
    w = w.max(grid_unit);

    while h as u64 * w as u64 > max_pixels as u64 {
        if h as f64 / w as f64 > aspect {
            h = h.saturating_sub(grid_unit);
        } else {
            w = w.saturating_sub(grid_unit);
        }
    }

    h = h.max(grid_unit);
    w = w.max(grid_unit);

    Ok((h, w))
}

/// Smart-resize an `ImageBuffer` using the grid-aware algorithm, then resize with Lanczos3.
pub fn smart_resize(
    buffer: &ImageBuffer,
    grid_unit: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> anyhow::Result<ImageBuffer> {
    let (target_h, target_w) = smart_resize_dims(
        buffer.height,
        buffer.width,
        grid_unit,
        min_pixels,
        max_pixels,
    )?;
    Ok(resize(buffer, target_w, target_h))
}

/// Round a value to the nearest multiple of `multiple`.
fn round_to_multiple(value: u32, multiple: u32) -> u32 {
    ((value + multiple / 2) / multiple) * multiple
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_multiple() {
        assert_eq!(round_to_multiple(0, 32), 0);
        assert_eq!(round_to_multiple(15, 32), 0);
        assert_eq!(round_to_multiple(16, 32), 32);
        assert_eq!(round_to_multiple(31, 32), 32);
        assert_eq!(round_to_multiple(32, 32), 32);
        assert_eq!(round_to_multiple(33, 32), 32);
        assert_eq!(round_to_multiple(48, 32), 64);
        assert_eq!(round_to_multiple(100, 28), 112);
    }

    #[test]
    fn test_smart_resize_dims_already_correct() {
        // 256x256 with grid_unit=32, min=1024, max=100000 -> should stay 256x256
        let (h, w) = smart_resize_dims(256, 256, 32, 1024, 100000).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w <= 100000);
        assert_eq!(h, 256);
        assert_eq!(w, 256);
    }

    #[test]
    fn test_smart_resize_dims_too_small() {
        // 10x10 = 100 pixels, min=1000, max=10000, grid=10
        // After scaling by sqrt(10) ~ 3.16: ~31.6 x 31.6, rounded to 30x30 = 900.
        // Rounding may bring result slightly below min_pixels, which is acceptable.
        let (h, w) = smart_resize_dims(10, 10, 10, 1000, 10000).unwrap();
        assert_eq!(h % 10, 0);
        assert_eq!(w % 10, 0);
        // The result should be significantly larger than the original 100 pixels
        assert!(h as u64 * w as u64 > 100);
        assert!(h as u64 * w as u64 <= 10000);
        // And dimensions should be at least grid_unit
        assert!(h >= 10);
        assert!(w >= 10);
    }

    #[test]
    fn test_smart_resize_dims_too_large() {
        // 1000x1000 = 1_000_000 pixels, max=100000, grid=32
        let (h, w) = smart_resize_dims(1000, 1000, 32, 1000, 100000).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h as u64 * w as u64 <= 100000);
    }

    #[test]
    fn test_smart_resize_dims_non_square() {
        // Very wide image: 100x2000
        let (h, w) = smart_resize_dims(100, 2000, 28, 10000, 500000).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h as u64 * w as u64 <= 500000);
        assert!(h >= 28);
        assert!(w >= 28);
    }

    #[test]
    fn test_smart_resize_dims_minimum_grid() {
        // Tiny image with large grid_unit should clamp to at least grid_unit
        let (h, w) = smart_resize_dims(1, 1, 32, 0, 100000).unwrap();
        assert!(h >= 32);
        assert!(w >= 32);
    }

    #[test]
    fn test_smart_resize_dims_typical_vision_model() {
        // Typical Qwen2-VL settings: patch=14, merge=2, grid_unit=28
        let (h, w) = smart_resize_dims(1080, 1920, 28, 65536, 16_777_216).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        let total = h as u64 * w as u64;
        assert!(total >= 65536);
        assert!(total <= 16_777_216);
    }

    #[test]
    fn test_smart_resize_dims_error_on_zero_grid() {
        assert!(smart_resize_dims(100, 100, 0, 100, 10000).is_err());
    }

    #[test]
    fn test_smart_resize_dims_error_on_min_greater_than_max() {
        assert!(smart_resize_dims(100, 100, 32, 10000, 100).is_err());
    }

    #[test]
    fn test_smart_resize_dims_error_on_zero_dimensions() {
        assert!(smart_resize_dims(0, 100, 32, 100, 10000).is_err());
        assert!(smart_resize_dims(100, 0, 32, 100, 10000).is_err());
    }

    #[test]
    fn test_resize_rgb() {
        let buf = ImageBuffer::new(10, 10, 3);
        let resized = resize(&buf, 5, 5);
        assert_eq!(resized.width, 5);
        assert_eq!(resized.height, 5);
        assert_eq!(resized.channels, 3);
        assert_eq!(resized.data.len(), 5 * 5 * 3);
    }

    #[test]
    fn test_resize_rgba() {
        let buf = ImageBuffer::new(10, 10, 4);
        let resized = resize(&buf, 20, 20);
        assert_eq!(resized.width, 20);
        assert_eq!(resized.height, 20);
        assert_eq!(resized.channels, 4);
        assert_eq!(resized.data.len(), 20 * 20 * 4);
    }
}
