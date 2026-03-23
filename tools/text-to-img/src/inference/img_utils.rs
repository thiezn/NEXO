//! Image decoding and preprocessing utilities for img2img.

use anyhow::Result;
use local_inference_helpers::candle_core::{DType, Device, Tensor};

/// Normalization range for source images before VAE encoding.
pub enum NormalizeRange {
    /// [0, 1] for flow-matching models (FLUX, Z-Image, Flux.2, Qwen-Image).
    ZeroToOne,
}

/// Decode PNG/JPEG bytes into a [1, 3, H, W] tensor normalized to the specified range,
/// resized to target dimensions.
pub fn decode_source_image(
    bytes: &[u8],
    target_w: u32,
    target_h: u32,
    range: NormalizeRange,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;

    let img = img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();

    let data: Vec<f32> = raw.iter().map(|&v| v as f32 / 255.0).collect();

    let tensor = Tensor::from_vec(data, (h, w, 3), &Device::Cpu)?;
    let tensor = tensor.permute((2, 0, 1))?; // [3, H, W]

    let tensor = match range {
        NormalizeRange::ZeroToOne => tensor,
    };

    let tensor = tensor.unsqueeze(0)?; // [1, 3, H, W]
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

/// Decode a mask image (PNG/JPEG) into a [1, 1, latent_h, latent_w] tensor with values in [0, 1].
/// White (255) = 1.0 = repaint region, Black (0) = 0.0 = preserve region.
pub fn decode_mask_image(
    bytes: &[u8],
    latent_height: usize,
    latent_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode mask image: {e}"))?;

    let img = img.resize_exact(
        latent_width as u32,
        latent_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let gray = img.to_luma8();

    let data: Vec<f32> = gray.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    let tensor = Tensor::from_vec(data, (1, 1, latent_height, latent_width), &Device::Cpu)?;
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

/// Context for inpainting: holds pre-computed tensors needed during the denoising loop.
pub struct InpaintContext {
    /// VAE-encoded original latents (unnoised).
    pub original_latents: Tensor,
    /// Mask tensor [1, 1, latent_h, latent_w] with values in [0, 1].
    pub mask: Tensor,
    /// Noise tensor matching latent shape, for re-noising the original at each step.
    pub noise: Tensor,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_source_image_shape() {
        let png = tiny_png();
        let tensor =
            decode_source_image(&png, 8, 8, NormalizeRange::ZeroToOne, &Device::Cpu, DType::F32)
                .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 8, 8]);
    }

    #[test]
    fn decode_source_image_zero_to_one_range() {
        let png = tiny_png();
        let tensor =
            decode_source_image(&png, 4, 4, NormalizeRange::ZeroToOne, &Device::Cpu, DType::F32)
                .unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min >= -0.01);
        assert!(max <= 1.01);
    }

    #[test]
    fn decode_source_image_resize() {
        let png = tiny_png();
        let tensor =
            decode_source_image(&png, 16, 16, NormalizeRange::ZeroToOne, &Device::Cpu, DType::F32)
                .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 16, 16]);
    }

    fn white_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    fn black_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_mask_shape() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 8, 8, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 8, 8]);
    }

    #[test]
    fn decode_mask_white_is_one() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min > 0.99);
    }

    #[test]
    fn decode_mask_black_is_zero() {
        let mask = black_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max < 0.01);
    }
}
