use anyhow::{bail, Result};
use local_inference_helpers::candle_core::Tensor;

use crate::models::OutputFormat;

/// Encode a candle tensor [3, H, W] of u8 values into PNG or JPEG bytes.
pub fn encode_image(img: &Tensor, format: OutputFormat, width: u32, height: u32) -> Result<Vec<u8>> {
    let (c, _h, _w) = img.dims3()?;
    if c != 3 {
        bail!("expected 3 channels, got {c}");
    }

    let img_data = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let rgb_image = image::RgbImage::from_raw(width, height, img_data)
        .ok_or_else(|| anyhow::anyhow!("failed to create image from tensor data"))?;

    let mut buf = std::io::Cursor::new(Vec::new());
    match format {
        OutputFormat::Png => rgb_image.write_to(&mut buf, image::ImageFormat::Png)?,
        OutputFormat::Jpeg => rgb_image.write_to(&mut buf, image::ImageFormat::Jpeg)?,
    }

    Ok(buf.into_inner())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use local_inference_helpers::candle_core::{DType, Device, Tensor};

    fn solid_red_tensor(h: usize, w: usize) -> Tensor {
        let mut data = vec![0u8; 3 * h * w];
        for i in 0..(h * w) {
            data[i] = 255;
        }
        Tensor::from_vec(data, (3, h, w), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
    }

    #[test]
    fn encode_png_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4).unwrap();
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn encode_jpeg_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4).unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_wrong_channels_fails() {
        let data = vec![0u8; 4 * 4 * 4];
        let tensor = Tensor::from_vec(data, (4, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let result = encode_image(&tensor, OutputFormat::Png, 4, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected 3 channels"));
    }

    #[test]
    fn encode_both_formats_differ() {
        let tensor = solid_red_tensor(4, 4);
        let png = encode_image(&tensor, OutputFormat::Png, 4, 4).unwrap();
        let jpeg = encode_image(&tensor, OutputFormat::Jpeg, 4, 4).unwrap();
        assert_ne!(png, jpeg);
    }
}
