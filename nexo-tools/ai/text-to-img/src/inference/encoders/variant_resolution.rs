//! Shared T5 and Qwen3 encoder variant resolution logic.

use anyhow::{bail, Result};
use local_inference_helpers::candle_core::Device;
use local_inference_helpers::device::{fits_in_memory, fmt_gb, should_use_gpu};
use local_inference_helpers::download::{cached_file_path, download_single_file_sync};
use local_inference_helpers::progress::ProgressReporter;
use std::path::{Path, PathBuf};

use crate::manifest::{
    find_qwen3_variant, find_t5_variant, known_qwen3_variants, known_t5_variants, T5_FP16_SIZE,
};

/// VRAM thresholds for text encoder placement decisions.
/// T5: FP16 needs ~10.5GB effective (9.2GB model + overhead).
const T5_VRAM_THRESHOLD: u64 = 10_500_000_000;
/// Qwen3: BF16 needs ~9.5GB effective (8.2GB model + overhead).
const QWEN3_FP16_VRAM_THRESHOLD: u64 = 9_500_000_000;

fn t5_vram_threshold(size_bytes: u64) -> u64 {
    size_bytes + 1_300_000_000 // model + ~1.3GB overhead
}

fn qwen3_vram_threshold(size_bytes: u64) -> u64 {
    size_bytes + 1_300_000_000
}

/// Resolve which T5 encoder variant to use and where to place it.
///
/// Returns `(encoder_path, on_gpu, device_label)`.
pub(crate) fn resolve_t5_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    default_t5_path: &Path,
) -> Result<(PathBuf, bool, String)> {
    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();

    match preference {
        Some(tag) if tag != "fp16" && tag != "auto" => {
            let variant = find_t5_variant(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown T5 variant '{tag}'. Valid: fp16, auto, q8, q6, q5, q4, q3",
                )
            })?;
            let path = resolve_t5_gguf_path(progress, variant)?;
            let threshold = t5_vram_threshold(variant.size_bytes);
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
            let label = if on_gpu {
                "GPU, quantized"
            } else {
                "CPU, quantized"
            };
            progress.info(&format!(
                "Using T5 {} ({}) on {}",
                variant.tag,
                fmt_gb(variant.size_bytes),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            Ok((path, on_gpu, label.to_string()))
        }

        Some("fp16") => {
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD);
            let label = if on_gpu { "GPU" } else { "CPU" };
            progress.info(&format!("Using FP16 T5 on {label} (explicit)"));
            Ok((default_t5_path.to_path_buf(), on_gpu, label.to_string()))
        }

        _ => {
            // Auto: try FP16 on GPU, then quantized on GPU, then FP16 on CPU
            if fits_in_memory(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD) {
                progress.info("Loading FP16 T5 on GPU");
                return Ok((default_t5_path.to_path_buf(), true, "GPU".to_string()));
            }

            if is_cuda || is_metal {
                for variant in known_t5_variants() {
                    let threshold = t5_vram_threshold(variant.size_bytes);
                    if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                        let path = resolve_or_download_t5(progress, variant)?;
                        progress.info(&format!(
                            "FP16 T5 ({}) exceeds VRAM ({}). Using T5 {} ({}) on GPU.",
                            fmt_gb(T5_FP16_SIZE),
                            fmt_gb(free_vram),
                            variant.tag,
                            fmt_gb(variant.size_bytes),
                        ));
                        return Ok((path, true, format!("GPU, quantized {}", variant.tag)));
                    }
                }
            }

            // On Metal, never fall back to CPU — use smallest quantized variant
            if is_metal {
                if let Some(smallest) = known_t5_variants().last() {
                    let path = resolve_t5_gguf_path(progress, smallest)?;
                    progress.info(&format!(
                        "Memory tight — using smallest T5 {} on GPU",
                        smallest.tag,
                    ));
                    return Ok((path, true, format!("GPU, quantized {}", smallest.tag)));
                }
            }

            progress.info("Loading FP16 T5 on CPU");
            Ok((default_t5_path.to_path_buf(), false, "CPU".to_string()))
        }
    }
}

fn resolve_t5_gguf_path(
    progress: &ProgressReporter,
    variant: &crate::manifest::T5Variant,
) -> Result<PathBuf> {
    if let Some(path) =
        cached_file_path(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
    {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading T5 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
        .map_err(|e| anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag))
}

fn resolve_or_download_t5(
    progress: &ProgressReporter,
    variant: &crate::manifest::T5Variant,
) -> Result<PathBuf> {
    if let Some(path) =
        cached_file_path(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
    {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading T5 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
        .map_err(|e| anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag))
}

/// Resolve which Qwen3 encoder variant to use.
///
/// Returns `(encoder_paths, is_gguf, on_gpu, device_label)`.
pub(crate) fn resolve_qwen3_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    bf16_paths: &[PathBuf],
    have_bf16: bool,
    prefer_gguf: bool,
) -> Result<(Vec<PathBuf>, bool, bool, String)> {
    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();

    match preference {
        Some(tag) if tag != "bf16" && tag != "auto" => {
            let variant = find_qwen3_variant(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown Qwen3 variant '{tag}'. Valid: bf16, auto, q8, q6, iq4, q3",
                )
            })?;
            let path = resolve_qwen3_gguf_path(progress, variant)?;
            let threshold = qwen3_vram_threshold(variant.size_bytes);
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
            let label = if on_gpu {
                "GPU, quantized"
            } else {
                "CPU, quantized"
            };
            progress.info(&format!(
                "Using Qwen3 {} ({}) on {}",
                variant.tag,
                fmt_gb(variant.size_bytes),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            Ok((vec![path], true, on_gpu, label.to_string()))
        }

        Some("bf16") => {
            if !have_bf16 {
                bail!(
                    "BF16 Qwen3 requested but shard files are missing. \
                     Run `text_to_img pull` for a model with Qwen3 or use a quantized variant."
                );
            }
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, QWEN3_FP16_VRAM_THRESHOLD);
            let label = if on_gpu { "GPU" } else { "CPU" };
            progress.info(&format!("Using BF16 Qwen3 on {label} (explicit)"));
            Ok((bf16_paths.to_vec(), false, on_gpu, label.to_string()))
        }

        _ => {
            if prefer_gguf {
                // Flux.2 path: prefer GGUF for multi-layer extraction
                return resolve_qwen3_auto_gguf(
                    progress, gpu_device, free_vram, bf16_paths, have_bf16, is_cuda, is_metal,
                );
            }

            // Z-Image path: try BF16 on GPU first, then quantized, then BF16 on CPU
            if have_bf16
                && fits_in_memory(is_cuda, is_metal, free_vram, QWEN3_FP16_VRAM_THRESHOLD)
            {
                progress.info("Loading BF16 Qwen3 on GPU");
                return Ok((bf16_paths.to_vec(), false, true, "GPU".to_string()));
            }

            // BF16 won't fit — try quantized variants
            if is_cuda || is_metal || !have_bf16 {
                for variant in known_qwen3_variants() {
                    let threshold = qwen3_vram_threshold(variant.size_bytes);
                    if fits_in_memory(is_cuda, is_metal, free_vram, threshold)
                        || (!is_cuda && !is_metal)
                    {
                        let path = resolve_or_download_qwen3(progress, variant)?;
                        let on_gpu = is_cuda || is_metal;
                        progress.info(&format!(
                            "Using Qwen3 {} ({}) on {}",
                            variant.tag,
                            fmt_gb(variant.size_bytes),
                            if on_gpu { "GPU" } else { "CPU" },
                        ));
                        return Ok((
                            vec![path],
                            true,
                            on_gpu,
                            format!(
                                "{}, quantized {}",
                                if on_gpu { "GPU" } else { "CPU" },
                                variant.tag,
                            ),
                        ));
                    }
                }
            }

            // Metal: use smallest quantized on GPU
            if is_metal {
                if let Some(smallest) = known_qwen3_variants().last() {
                    let path = resolve_or_download_qwen3(progress, smallest)?;
                    progress.info(&format!(
                        "Memory tight — using smallest Qwen3 {} on GPU",
                        smallest.tag,
                    ));
                    return Ok((
                        vec![path],
                        true,
                        true,
                        format!("GPU, quantized {}", smallest.tag),
                    ));
                }
            }

            if have_bf16 {
                progress.info("Loading BF16 Qwen3 on CPU");
                return Ok((bf16_paths.to_vec(), false, false, "CPU".to_string()));
            }

            bail!(
                "No Qwen3 text encoder available. Run `text_to_img pull` for a model with Qwen3."
            );
        }
    }
}

fn resolve_qwen3_auto_gguf(
    progress: &ProgressReporter,
    _gpu_device: &Device,
    free_vram: u64,
    bf16_paths: &[PathBuf],
    have_bf16: bool,
    is_cuda: bool,
    is_metal: bool,
) -> Result<(Vec<PathBuf>, bool, bool, String)> {
    if is_cuda || is_metal {
        for variant in known_qwen3_variants() {
            let threshold = qwen3_vram_threshold(variant.size_bytes);
            if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                let path = resolve_or_download_qwen3(progress, variant)?;
                progress.info(&format!(
                    "Using quantized Qwen3 {} ({}) on GPU",
                    variant.tag,
                    fmt_gb(variant.size_bytes),
                ));
                return Ok((
                    vec![path],
                    true,
                    true,
                    format!("GPU, quantized {}", variant.tag),
                ));
            }
        }
    }

    if have_bf16 {
        progress.info("Loading BF16 Qwen3 on CPU");
        Ok((bf16_paths.to_vec(), false, false, "CPU".to_string()))
    } else {
        bail!("No Qwen3 encoder available (no BF16 files and no GGUF cached)")
    }
}

fn resolve_qwen3_gguf_path(
    progress: &ProgressReporter,
    variant: &crate::manifest::Qwen3Variant,
) -> Result<PathBuf> {
    if let Some(path) = cached_file_path(
        variant.hf_repo,
        variant.hf_filename,
        Some("shared/qwen3-gguf"),
    ) {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading Qwen3 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(
        variant.hf_repo,
        variant.hf_filename,
        Some("shared/qwen3-gguf"),
    )
    .map_err(|e| anyhow::anyhow!("failed to download Qwen3 {}: {e}", variant.tag))
}

fn resolve_or_download_qwen3(
    progress: &ProgressReporter,
    variant: &crate::manifest::Qwen3Variant,
) -> Result<PathBuf> {
    resolve_qwen3_gguf_path(progress, variant)
}
