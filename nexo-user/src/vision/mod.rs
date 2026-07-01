pub mod decode;
pub mod display;
pub mod encode;
pub mod normalize;
pub mod resize;

pub use decode::{load_bytes, load_file};
pub use display::{DisplayConfig, display_in_terminal};
pub use encode::{encode_jpeg, encode_png, save_file};
pub use normalize::{
    NormalizeConfig, from_rgb_f32, from_rgb_f32_hwc, normalize_rgb_f32, to_rgb_f32,
};
pub use resize::{resize, smart_resize, smart_resize_dims};

/// A simple image buffer holding raw pixel data in row-major order.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    /// Raw pixel data in RGB8 or RGBA8 row-major layout.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Number of channels: 3 for RGB, 4 for RGBA.
    pub channels: u8,
}

impl ImageBuffer {
    /// Create a new `ImageBuffer` with the given dimensions and channel count.
    /// The data vector is initialized to all zeros.
    pub fn new(width: u32, height: u32, channels: u8) -> Self {
        let size = width as usize * height as usize * channels as usize;
        Self {
            data: vec![0u8; size],
            width,
            height,
            channels,
        }
    }

    /// Create an `ImageBuffer` from existing RGB8 data.
    ///
    /// Returns an error if the data length does not match `width * height * 3`.
    pub fn from_rgb(data: Vec<u8>, width: u32, height: u32) -> anyhow::Result<Self> {
        let expected = width as usize * height as usize * 3;
        anyhow::ensure!(
            data.len() == expected,
            "RGB data length {} does not match expected {} ({}x{}x3)",
            data.len(),
            expected,
            width,
            height,
        );
        Ok(Self {
            data,
            width,
            height,
            channels: 3,
        })
    }

    /// Create an `ImageBuffer` from existing RGBA8 data.
    ///
    /// Returns an error if the data length does not match `width * height * 4`.
    pub fn from_rgba(data: Vec<u8>, width: u32, height: u32) -> anyhow::Result<Self> {
        let expected = width as usize * height as usize * 4;
        anyhow::ensure!(
            data.len() == expected,
            "RGBA data length {} does not match expected {} ({}x{}x4)",
            data.len(),
            expected,
            width,
            height,
        );
        Ok(Self {
            data,
            width,
            height,
            channels: 4,
        })
    }

    /// Return the total number of pixels in the image.
    pub fn num_pixels(&self) -> u32 {
        self.width * self.height
    }

    /// Convert this buffer to RGB8. If already RGB, returns a clone.
    /// If RGBA, strips the alpha channel.
    pub fn to_rgb(&self) -> Self {
        if self.channels == 3 {
            return self.clone();
        }

        let pixel_count = self.width as usize * self.height as usize;
        let mut rgb_data = Vec::with_capacity(pixel_count * 3);
        for pixel in self.data.chunks_exact(4) {
            rgb_data.push(pixel[0]);
            rgb_data.push(pixel[1]);
            rgb_data.push(pixel[2]);
        }

        Self {
            data: rgb_data,
            width: self.width,
            height: self.height,
            channels: 3,
        }
    }

    /// Convert this buffer to RGBA8. If already RGBA, returns a clone.
    /// If RGB, adds a fully opaque alpha channel (255).
    pub fn to_rgba(&self) -> Self {
        if self.channels == 4 {
            return self.clone();
        }

        let pixel_count = self.width as usize * self.height as usize;
        let mut rgba_data = Vec::with_capacity(pixel_count * 4);
        for pixel in self.data.chunks_exact(3) {
            rgba_data.push(pixel[0]);
            rgba_data.push(pixel[1]);
            rgba_data.push(pixel[2]);
            rgba_data.push(255);
        }

        Self {
            data: rgba_data,
            width: self.width,
            height: self.height,
            channels: 4,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buf = ImageBuffer::new(10, 20, 3);
        assert_eq!(buf.width, 10);
        assert_eq!(buf.height, 20);
        assert_eq!(buf.channels, 3);
        assert_eq!(buf.data.len(), 10 * 20 * 3);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_new_rgba() {
        let buf = ImageBuffer::new(5, 5, 4);
        assert_eq!(buf.data.len(), 5 * 5 * 4);
    }

    #[test]
    fn test_from_rgb() {
        let data = vec![128u8; 4 * 4 * 3];
        let buf = ImageBuffer::from_rgb(data, 4, 4).unwrap();
        assert_eq!(buf.channels, 3);
        assert_eq!(buf.num_pixels(), 16);
    }

    #[test]
    fn test_from_rgb_wrong_size() {
        let data = vec![0u8; 10];
        assert!(ImageBuffer::from_rgb(data, 4, 4).is_err());
    }

    #[test]
    fn test_from_rgba() {
        let data = vec![200u8; 3 * 3 * 4];
        let buf = ImageBuffer::from_rgba(data, 3, 3).unwrap();
        assert_eq!(buf.channels, 4);
        assert_eq!(buf.num_pixels(), 9);
    }

    #[test]
    fn test_from_rgba_wrong_size() {
        let data = vec![0u8; 5];
        assert!(ImageBuffer::from_rgba(data, 2, 2).is_err());
    }

    #[test]
    fn test_num_pixels() {
        let buf = ImageBuffer::new(100, 200, 3);
        assert_eq!(buf.num_pixels(), 20000);
    }

    #[test]
    fn test_to_rgb_noop() {
        let data = vec![42u8; 2 * 2 * 3];
        let buf = ImageBuffer::from_rgb(data.clone(), 2, 2).unwrap();
        let rgb = buf.to_rgb();
        assert_eq!(rgb.channels, 3);
        assert_eq!(rgb.data, data);
    }

    #[test]
    fn test_to_rgb_from_rgba() {
        // Single pixel: R=10, G=20, B=30, A=255
        let data = vec![10, 20, 30, 255];
        let buf = ImageBuffer::from_rgba(data, 1, 1).unwrap();
        let rgb = buf.to_rgb();
        assert_eq!(rgb.channels, 3);
        assert_eq!(rgb.data, vec![10, 20, 30]);
    }

    #[test]
    fn test_to_rgba_noop() {
        let data = vec![42u8; 2 * 2 * 4];
        let buf = ImageBuffer::from_rgba(data.clone(), 2, 2).unwrap();
        let rgba = buf.to_rgba();
        assert_eq!(rgba.channels, 4);
        assert_eq!(rgba.data, data);
    }

    #[test]
    fn test_to_rgba_from_rgb() {
        // Single pixel: R=10, G=20, B=30
        let data = vec![10, 20, 30];
        let buf = ImageBuffer::from_rgb(data, 1, 1).unwrap();
        let rgba = buf.to_rgba();
        assert_eq!(rgba.channels, 4);
        assert_eq!(rgba.data, vec![10, 20, 30, 255]);
    }

    #[test]
    fn test_roundtrip_rgb_rgba_rgb() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let buf = ImageBuffer::from_rgb(data.clone(), 2, 2).unwrap();
        let rgba = buf.to_rgba();
        let rgb = rgba.to_rgb();
        assert_eq!(rgb.data, data);
        assert_eq!(rgb.channels, 3);
    }
}
