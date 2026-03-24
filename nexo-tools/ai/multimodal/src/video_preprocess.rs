#[cfg(feature = "video")]
mod inner {
    use anyhow::Context;
    use std::path::Path;
    use video_rs::Decoder;

    use crate::image_preprocess::{ImagePreprocessor, smart_resize};

    pub struct PreprocessedVideo {
        pub pixel_data: Vec<f32>,
        pub total_patches: usize,
        pub patch_dim: usize,
        pub grid_thw: [u32; 3],
        pub num_video_tokens: usize,
    }

    pub fn preprocess_video(
        video_path: &Path,
        sample_fps: f64,
        config: &ImagePreprocessor,
    ) -> anyhow::Result<PreprocessedVideo> {
        video_rs::init();

        let mut decoder = Decoder::new(video_path)
            .with_context(|| format!("failed to open video: {}", video_path.display()))?;

        let (src_w, src_h) = decoder.size();
        let src_fps = decoder.frame_rate();
        let frame_interval = (src_fps / sample_fps).max(1.0) as usize;

        tracing::info!(
            width = src_w,
            height = src_h,
            fps = src_fps,
            sample_fps,
            frame_interval,
            "decoding video"
        );

        let (target_h, target_w) = smart_resize(
            src_h as usize,
            src_w as usize,
            config.grid_unit(),
            config.min_pixels,
            config.max_pixels,
        )?;

        tracing::info!(target_width = target_w, target_height = target_h, "video frame dimensions");

        let ps = config.patch_size;
        let tps = config.temporal_patch_size;
        let ms = config.spatial_merge_size;
        let h_patches = target_h / ps;
        let w_patches = target_w / ps;
        let single_frame = ps * ps;
        let patch_dim = 3 * tps * single_frame;

        let mut frames: Vec<image::RgbImage> = Vec::new();
        let mut frame_idx = 0usize;

        while let Some(Ok((_ts, frame))) = decoder.decode_iter().next() {
            if frame_idx % frame_interval == 0 {
                let (fh, fw, _) = frame.dim();
                let raw: Vec<u8> = frame.into_raw_vec_and_offset().0;
                let img = image::RgbImage::from_raw(fw as u32, fh as u32, raw)
                    .ok_or_else(|| anyhow::anyhow!("failed to create image from video frame"))?;
                let resized = image::imageops::resize(
                    &img,
                    target_w as u32,
                    target_h as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                frames.push(resized);
            }
            frame_idx += 1;
        }

        if frames.is_empty() {
            anyhow::bail!("no frames decoded from video");
        }

        while frames.len() % tps != 0 {
            frames.push(frames.last().unwrap().clone());
        }

        let num_temporal_patches = frames.len() / tps;
        let total_patches = num_temporal_patches * h_patches * w_patches;

        tracing::info!(
            decoded_frames = frames.len(),
            temporal_patches = num_temporal_patches,
            total_patches,
            "video frames preprocessed"
        );

        let mut patch_data = vec![0f32; total_patches * patch_dim];

        for tp in 0..num_temporal_patches {
            for ph in 0..h_patches {
                for pw in 0..w_patches {
                    let patch_idx = tp * h_patches * w_patches + ph * w_patches + pw;
                    let base = patch_idx * patch_dim;

                    for c in 0..3usize {
                        for t in 0..tps {
                            let frame = &frames[tp * tps + t];
                            let pixels = frame.as_raw();
                            for py in 0..ps {
                                for px in 0..ps {
                                    let src_y = ph * ps + py;
                                    let src_x = pw * ps + px;
                                    let val =
                                        pixels[(src_y * target_w + src_x) * 3 + c] as f32 / 255.0;
                                    let normalized =
                                        (val - config.image_mean[c]) / config.image_std[c];
                                    let offset =
                                        c * (tps * single_frame) + t * single_frame + py * ps + px;
                                    patch_data[base + offset] = normalized;
                                }
                            }
                        }
                    }
                }
            }
        }

        let num_video_tokens = num_temporal_patches * (h_patches / ms) * (w_patches / ms);

        Ok(PreprocessedVideo {
            pixel_data: patch_data,
            total_patches,
            patch_dim,
            grid_thw: [num_temporal_patches as u32, h_patches as u32, w_patches as u32],
            num_video_tokens,
        })
    }
}

#[cfg(feature = "video")]
pub use inner::*;
