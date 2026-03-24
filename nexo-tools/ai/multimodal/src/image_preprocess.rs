use anyhow::Context;
use std::path::Path;

const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

pub struct ImagePreprocessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65536,
            max_pixels: 16_777_216,
            image_mean: IMAGE_MEAN,
            image_std: IMAGE_STD,
        }
    }
}

impl ImagePreprocessor {
    pub fn from_config_file(path: Option<&Path>) -> anyhow::Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        if !path.exists() {
            return Ok(Self::default());
        }

        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading preprocessor config: {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&data)?;

        let mut config = Self::default();

        if let Some(size) = json.get("size") {
            if let Some(v) = size.get("shortest_edge").and_then(|v| v.as_u64()) {
                config.min_pixels = v as usize;
            }
            if let Some(v) = size.get("longest_edge").and_then(|v| v.as_u64()) {
                config.max_pixels = v as usize;
            }
        }
        if let Some(v) = json.get("min_pixels").and_then(|v| v.as_u64()) {
            config.min_pixels = v as usize;
        }
        if let Some(v) = json.get("max_pixels").and_then(|v| v.as_u64()) {
            config.max_pixels = v as usize;
        }
        if let Some(v) = json.get("patch_size").and_then(|v| v.as_u64()) {
            config.patch_size = v as usize;
        }
        if let Some(v) = json.get("merge_size").and_then(|v| v.as_u64()) {
            config.spatial_merge_size = v as usize;
        }
        if let Some(v) = json.get("temporal_patch_size").and_then(|v| v.as_u64()) {
            config.temporal_patch_size = v as usize;
        }
        if let Some(arr) = json.get("image_mean").and_then(|v| v.as_array())
            && arr.len() == 3
        {
            for (i, val) in arr.iter().enumerate() {
                if let Some(f) = val.as_f64() {
                    config.image_mean[i] = f as f32;
                }
            }
        }
        if let Some(arr) = json.get("image_std").and_then(|v| v.as_array())
            && arr.len() == 3
        {
            for (i, val) in arr.iter().enumerate() {
                if let Some(f) = val.as_f64() {
                    config.image_std[i] = f as f32;
                }
            }
        }

        Ok(config)
    }

    pub fn grid_unit(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }
}

/// Framework-agnostic preprocessed image data.
pub struct PreprocessedImage {
    /// Flat pixel data: shape (total_patches, C * temporal_patch_size * patch_size * patch_size)
    pub pixel_data: Vec<f32>,
    pub total_patches: usize,
    pub patch_dim: usize,
    /// Grid: [t, h_patches, w_patches]
    pub grid_thw: [u32; 3],
    pub num_image_tokens: usize,
}

pub fn preprocess_image(
    image_path: &Path,
    config: &ImagePreprocessor,
) -> anyhow::Result<PreprocessedImage> {
    let img = image::open(image_path)
        .with_context(|| format!("failed to open image: {}", image_path.display()))?
        .into_rgb8();

    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    tracing::info!(width = orig_w, height = orig_h, "loaded image");

    let (target_h, target_w) = smart_resize(
        orig_h,
        orig_w,
        config.grid_unit(),
        config.min_pixels,
        config.max_pixels,
    )?;

    tracing::info!(target_width = target_w, target_height = target_h, "resized dimensions");

    let resized = image::imageops::resize(
        &img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let pixels = resized.as_raw();
    let ps = config.patch_size;
    let tps = config.temporal_patch_size;
    let h_patches = target_h / ps;
    let w_patches = target_w / ps;
    let total_patches = h_patches * w_patches;
    let patch_dim = 3 * tps * ps * ps;
    let single_frame = ps * ps;
    let mut patch_data = vec![0f32; total_patches * patch_dim];

    for ph in 0..h_patches {
        for pw in 0..w_patches {
            let base = (ph * w_patches + pw) * patch_dim;
            for c in 0..3usize {
                for py in 0..ps {
                    for px in 0..ps {
                        let src_y = ph * ps + py;
                        let src_x = pw * ps + px;
                        let val = pixels[(src_y * target_w + src_x) * 3 + c] as f32 / 255.0;
                        let normalized = (val - config.image_mean[c]) / config.image_std[c];
                        let spatial_offset = py * ps + px;
                        let channel_base = base + c * (tps * single_frame);
                        for t in 0..tps {
                            patch_data[channel_base + t * single_frame + spatial_offset] =
                                normalized;
                        }
                    }
                }
            }
        }
    }

    let ms = config.spatial_merge_size;
    let num_image_tokens = (h_patches / ms) * (w_patches / ms);

    tracing::info!(
        patches = total_patches,
        image_tokens = num_image_tokens,
        "image preprocessed"
    );

    Ok(PreprocessedImage {
        pixel_data: patch_data,
        total_patches,
        patch_dim,
        grid_thw: [1, h_patches as u32, w_patches as u32],
        num_image_tokens,
    })
}

pub fn smart_resize(
    orig_h: usize,
    orig_w: usize,
    grid_unit: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> anyhow::Result<(usize, usize)> {
    let aspect = orig_h as f64 / orig_w as f64;
    let total = orig_h * orig_w;

    let (mut h, mut w) = if total < min_pixels {
        let scale = (min_pixels as f64 / total as f64).sqrt();
        ((orig_h as f64 * scale) as usize, (orig_w as f64 * scale) as usize)
    } else if total > max_pixels {
        let scale = (max_pixels as f64 / total as f64).sqrt();
        ((orig_h as f64 * scale) as usize, (orig_w as f64 * scale) as usize)
    } else {
        (orig_h, orig_w)
    };

    h = round_to_multiple(h, grid_unit);
    w = round_to_multiple(w, grid_unit);

    h = h.max(grid_unit);
    w = w.max(grid_unit);

    while h * w > max_pixels {
        if h as f64 / w as f64 > aspect {
            h -= grid_unit;
        } else {
            w -= grid_unit;
        }
    }

    h = h.max(grid_unit);
    w = w.max(grid_unit);

    Ok((h, w))
}

fn round_to_multiple(value: usize, multiple: usize) -> usize {
    ((value + multiple / 2) / multiple) * multiple
}
