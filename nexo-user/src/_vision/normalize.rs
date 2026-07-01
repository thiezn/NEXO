use super::ImageBuffer;

/// Configuration for per-channel normalization: `(pixel / 255 - mean) / std`.
#[derive(Debug, Clone, Copy)]
pub struct NormalizeConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl NormalizeConfig {
    /// ImageNet normalization constants.
    pub const IMAGENET: Self = Self {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };

    /// Symmetric normalization mapping [0, 255] to [-1, 1].
    pub const SYMMETRIC: Self = Self {
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };
}

/// Normalize an RGB8 `ImageBuffer` to f32 values in CHW (channel-first) layout.
///
/// For each pixel at (x, y), the normalized value is:
/// `(pixel_u8 / 255.0 - mean[c]) / std[c]`
///
/// Output layout: `[C, H, W]` — channel index varies slowest.
/// Output length: `3 * width * height`.
pub fn normalize_rgb_f32(
    buffer: &ImageBuffer,
    config: &NormalizeConfig,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        buffer.channels == 3,
        "normalize_rgb_f32 requires an RGB buffer (3 channels), got {}",
        buffer.channels,
    );

    let w = buffer.width as usize;
    let h = buffer.height as usize;
    let hw = h * w;
    let mut out = vec![0.0f32; 3 * hw];

    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * 3;
            for c in 0..3 {
                let val = buffer.data[src_idx + c] as f32 / 255.0;
                let normalized = (val - config.mean[c]) / config.std[c];
                out[c * hw + y * w + x] = normalized;
            }
        }
    }

    Ok(out)
}

/// Convert an RGB8 `ImageBuffer` to f32 in CHW layout, scaled to [0.0, 1.0].
///
/// This is equivalent to normalization with mean=[0,0,0] and std=[1,1,1].
/// Output length: `3 * width * height`.
pub fn to_rgb_f32(buffer: &ImageBuffer) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        buffer.channels == 3,
        "to_rgb_f32 requires an RGB buffer (3 channels), got {}",
        buffer.channels,
    );

    let w = buffer.width as usize;
    let h = buffer.height as usize;
    let hw = h * w;
    let mut out = vec![0.0f32; 3 * hw];

    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * 3;
            for c in 0..3 {
                let val = buffer.data[src_idx + c] as f32 / 255.0;
                out[c * hw + y * w + x] = val;
            }
        }
    }

    Ok(out)
}

/// Convert f32 CHW data back to an RGB8 `ImageBuffer`.
///
/// Input layout: `[C, H, W]` with values in [0.0, 1.0].
/// Values are clamped to [0.0, 1.0] and scaled to [0, 255].
pub fn from_rgb_f32(data: &[f32], width: u32, height: u32) -> anyhow::Result<ImageBuffer> {
    let w = width as usize;
    let h = height as usize;
    let hw = h * w;

    anyhow::ensure!(
        data.len() == 3 * hw,
        "CHW data length {} does not match expected {} (3 * {} * {})",
        data.len(),
        3 * hw,
        height,
        width,
    );

    let mut rgb_data = vec![0u8; hw * 3];

    for y in 0..h {
        for x in 0..w {
            let dst_idx = (y * w + x) * 3;
            for c in 0..3 {
                let val = data[c * hw + y * w + x].clamp(0.0, 1.0);
                rgb_data[dst_idx + c] = (val * 255.0 + 0.5) as u8;
            }
        }
    }

    Ok(ImageBuffer {
        data: rgb_data,
        width,
        height,
        channels: 3,
    })
}

/// Convert f32 HWC data to an RGB8 `ImageBuffer`.
///
/// Input layout: `[H, W, C]` with values in [0.0, 1.0].
/// Values are clamped to [0.0, 1.0] and scaled to [0, 255].
pub fn from_rgb_f32_hwc(data: &[f32], width: u32, height: u32) -> anyhow::Result<ImageBuffer> {
    let w = width as usize;
    let h = height as usize;
    let total = h * w * 3;

    anyhow::ensure!(
        data.len() == total,
        "HWC data length {} does not match expected {} ({} * {} * 3)",
        data.len(),
        total,
        height,
        width,
    );

    let mut rgb_data = vec![0u8; total];

    for (i, &val) in data.iter().enumerate() {
        let clamped = val.clamp(0.0, 1.0);
        rgb_data[i] = (clamped * 255.0 + 0.5) as u8;
    }

    Ok(ImageBuffer {
        data: rgb_data,
        width,
        height,
        channels: 3,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_identity_roundtrip() {
        // Identity normalization: mean=0, std=1 means output = pixel / 255.0
        let identity = NormalizeConfig {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };

        // Create a 2x2 RGB image with known values
        let data = vec![
            0, 128, 255, // pixel (0,0)
            64, 192, 32, // pixel (1,0)
            100, 50, 200, // pixel (0,1)
            10, 20, 30, // pixel (1,1)
        ];
        let buf = ImageBuffer::from_rgb(data, 2, 2).unwrap();

        let normalized = normalize_rgb_f32(&buf, &identity).unwrap();
        assert_eq!(normalized.len(), 3 * 2 * 2);

        // Roundtrip back
        let restored = from_rgb_f32(&normalized, 2, 2).unwrap();
        assert_eq!(restored.channels, 3);
        assert_eq!(restored.width, 2);
        assert_eq!(restored.height, 2);

        // Check all values are close (rounding tolerance of 1)
        for (orig, rest) in buf.data.iter().zip(restored.data.iter()) {
            assert!(
                (*orig as i16 - *rest as i16).unsigned_abs() <= 1,
                "original={orig}, restored={rest}"
            );
        }
    }

    #[test]
    fn test_to_rgb_f32_and_back() {
        let data = vec![
            255, 0, 128, // pixel (0,0)
            0, 255, 64, // pixel (1,0)
        ];
        let buf = ImageBuffer::from_rgb(data.clone(), 2, 1).unwrap();

        let float_data = to_rgb_f32(&buf).unwrap();
        assert_eq!(float_data.len(), 3 * 2);

        // CHW layout check:
        // Channel 0 (R): [255/255, 0/255] = [1.0, 0.0]
        // Channel 1 (G): [0/255, 255/255] = [0.0, 1.0]
        // Channel 2 (B): [128/255, 64/255]
        assert!((float_data[0] - 1.0).abs() < 1e-5); // R of pixel 0
        assert!((float_data[1] - 0.0).abs() < 1e-5); // R of pixel 1
        assert!((float_data[2] - 0.0).abs() < 1e-5); // G of pixel 0
        assert!((float_data[3] - 1.0).abs() < 1e-5); // G of pixel 1

        // Roundtrip
        let restored = from_rgb_f32(&float_data, 2, 1).unwrap();
        for (orig, rest) in data.iter().zip(restored.data.iter()) {
            assert!(
                (*orig as i16 - *rest as i16).unsigned_abs() <= 1,
                "original={orig}, restored={rest}"
            );
        }
    }

    #[test]
    fn test_normalize_imagenet() {
        // Single white pixel: R=G=B=255
        let data = vec![255, 255, 255];
        let buf = ImageBuffer::from_rgb(data, 1, 1).unwrap();

        let normalized = normalize_rgb_f32(&buf, &NormalizeConfig::IMAGENET).unwrap();
        assert_eq!(normalized.len(), 3);

        // R: (1.0 - 0.485) / 0.229 = 2.2489...
        assert!((normalized[0] - (1.0 - 0.485) / 0.229).abs() < 1e-4);
        // G: (1.0 - 0.456) / 0.224
        assert!((normalized[1] - (1.0 - 0.456) / 0.224).abs() < 1e-4);
        // B: (1.0 - 0.406) / 0.225
        assert!((normalized[2] - (1.0 - 0.406) / 0.225).abs() < 1e-4);
    }

    #[test]
    fn test_normalize_symmetric() {
        // Black pixel: R=G=B=0
        let data = vec![0, 0, 0];
        let buf = ImageBuffer::from_rgb(data, 1, 1).unwrap();

        let normalized = normalize_rgb_f32(&buf, &NormalizeConfig::SYMMETRIC).unwrap();
        // (0.0 - 0.5) / 0.5 = -1.0
        for &v in &normalized {
            assert!((v - (-1.0)).abs() < 1e-5);
        }

        // White pixel: R=G=B=255
        let data = vec![255, 255, 255];
        let buf = ImageBuffer::from_rgb(data, 1, 1).unwrap();

        let normalized = normalize_rgb_f32(&buf, &NormalizeConfig::SYMMETRIC).unwrap();
        // (1.0 - 0.5) / 0.5 = 1.0
        for &v in &normalized {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_from_rgb_f32_clamps() {
        // 1x1 image in CHW layout: [R, G, B] = [-0.5, 1.5, 0.5]
        // Values outside [0, 1] should be clamped
        let data = vec![-0.5, 1.5, 0.5];
        let buf = from_rgb_f32(&data, 1, 1).unwrap();
        // -0.5 -> 0, 1.5 -> 255, 0.5 -> 128
        assert_eq!(buf.data[0], 0);
        assert_eq!(buf.data[1], 255);
        assert_eq!(buf.data[2], 128);
    }

    #[test]
    fn test_from_rgb_f32_wrong_size() {
        let data = vec![0.0; 10];
        assert!(from_rgb_f32(&data, 2, 2).is_err());
    }

    #[test]
    fn test_from_rgb_f32_hwc() {
        // 2x1 image in HWC layout: [R0, G0, B0, R1, G1, B1]
        let data = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.25];
        let buf = from_rgb_f32_hwc(&data, 2, 1).unwrap();
        assert_eq!(buf.channels, 3);
        assert_eq!(buf.width, 2);
        assert_eq!(buf.height, 1);
        assert_eq!(buf.data[0], 255); // R of pixel 0
        assert_eq!(buf.data[1], 0); // G of pixel 0
        assert_eq!(buf.data[2], 128); // B of pixel 0
        assert_eq!(buf.data[3], 0); // R of pixel 1
        assert_eq!(buf.data[4], 255); // G of pixel 1
        assert_eq!(buf.data[5], 64); // B of pixel 1
    }

    #[test]
    fn test_from_rgb_f32_hwc_wrong_size() {
        let data = vec![0.0; 5];
        assert!(from_rgb_f32_hwc(&data, 2, 2).is_err());
    }

    #[test]
    fn test_normalize_requires_rgb() {
        let buf = ImageBuffer::new(2, 2, 4);
        let config = NormalizeConfig::IMAGENET;
        assert!(normalize_rgb_f32(&buf, &config).is_err());
    }

    #[test]
    fn test_to_rgb_f32_requires_rgb() {
        let buf = ImageBuffer::new(2, 2, 4);
        assert!(to_rgb_f32(&buf).is_err());
    }
}
