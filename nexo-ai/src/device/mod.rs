use candle_core::{DType, Device};

/// Create a device for inference: Metal if available, CPU fallback.
///
/// The `on_info` callback receives informational messages about device selection.
/// Pass `|_| {}` to suppress output.
///
/// Override with env var `NEXO_AI_DEVICE=cpu` to force CPU inference.
pub fn create_device() -> anyhow::Result<Device> {
    let force_cpu = std::env::var("NEXO_AI_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);
    if force_cpu {
        tracing::warn!("CPU forced via NEXO_AI_DEVICE=cpu");
        return Ok(Device::Cpu);
    }
    if candle_core::utils::metal_is_available() {
        tracing::info!("Metal detected, using GPU");
        Ok(Device::new_metal(0)?)
    } else {
        tracing::warn!("No GPU detected, falling back to CPU");
        Ok(Device::Cpu)
    }
}

/// Dtype for image generation pipelines (diffusion denoising loops).
///
/// F32 on Metal — BF16 matmul accumulation errors compound through denoising
/// loops, causing washed-out blurry images. This matches mold's implementation.
/// F32 on CPU.
pub fn gpu_dtype(_device: &Device) -> DType {
    // Metal image gen needs F32 precision; BF16 causes quality issues
    DType::F32
}

/// Dtype for LLM / text model inference.
///
/// BF16 on Metal (Apple Silicon supports it natively, halves memory vs F32).
/// F32 on CPU.
pub fn gpu_compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    }
}

// ── macOS VM statistics FFI ─────────────────────────────────────────────────

#[cfg(target_os = "macos")]
struct MacOSMemInfo {
    free: u64,
    inactive: u64,
}

#[cfg(target_os = "macos")]
fn macos_vm_stats() -> Option<MacOSMemInfo> {
    type MachPort = u32;
    type KernReturn = i32;
    type HostFlavor = i32;
    type MachMsgType = u32;

    const HOST_VM_INFO64: HostFlavor = 4;
    const HOST_VM_INFO64_COUNT: MachMsgType = 38;
    const KERN_SUCCESS: KernReturn = 0;

    unsafe extern "C" {
        fn mach_host_self() -> MachPort;
        fn host_statistics64(
            host: MachPort,
            flavor: HostFlavor,
            info: *mut i32,
            count: *mut MachMsgType,
        ) -> KernReturn;
        fn host_page_size(host: MachPort, page_size: *mut usize) -> KernReturn;
    }

    unsafe {
        let mut buf = [0i32; HOST_VM_INFO64_COUNT as usize];
        let mut count = HOST_VM_INFO64_COUNT;
        let ret = host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            buf.as_mut_ptr(),
            &mut count,
        );
        if ret != KERN_SUCCESS {
            return None;
        }
        let mut page_size: usize = 0;
        let ret = host_page_size(mach_host_self(), &mut page_size);
        if ret != KERN_SUCCESS {
            return None;
        }
        let page_size = page_size as u64;
        Some(MacOSMemInfo {
            free: buf[0] as u32 as u64 * page_size,
            inactive: buf[2] as u32 as u64 * page_size,
        })
    }
}

// ── Memory query functions ──────────────────────────────────────────────────

/// Immediately free system memory on macOS (free pages only).
///
/// Conservative metric -- memory available WITHOUT reclaiming inactive pages.
#[cfg(target_os = "macos")]
pub fn free_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free)
}

/// Total available system memory on macOS (free + inactive pages).
///
/// Inactive pages CAN be reclaimed by the OS, but doing so involves I/O for dirty pages.
#[cfg(target_os = "macos")]
pub fn available_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free + s.inactive)
}

#[cfg(not(target_os = "macos"))]
pub fn free_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn available_system_memory_bytes() -> Option<u64> {
    None
}

// ── Decision functions ──────────────────────────────────────────────────────

/// Pre-flight memory guard: check if loading a component of `size_bytes` would
/// exhaust system memory.
///
/// - Hard-fails if the component would consume >90% of available memory.
/// - Warns if the component is >2x free memory (will require page reclamation).
/// - No-op on non-macOS platforms or if memory info is unavailable.
pub fn preflight_memory_check(component_name: &str, size_bytes: u64) -> anyhow::Result<()> {
    let available = match available_system_memory_bytes() {
        Some(a) if a > 0 => a,
        _ => return Ok(()),
    };

    if size_bytes > available * 90 / 100 {
        anyhow::bail!(
            "Not enough memory to load {} ({} needed, {} available).\n\
             Close other applications or use a smaller quantized model.",
            component_name,
            fmt_gb(size_bytes),
            fmt_gb(available),
        );
    }

    if size_bytes > available * 70 / 100 {
        tracing::warn!(
            "{} ({}) will use {:.0}% of available memory ({})",
            component_name,
            fmt_gb(size_bytes),
            size_bytes as f64 / available as f64 * 100.0,
            fmt_gb(available),
        );
    }

    Ok(())
}

/// Return a human-readable memory status string for display.
///
/// Example: `"Memory: 2.3 GB free, 8.1 GB inactive (10.4 GB available)"`
pub fn memory_status_string() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        if let Some(stats) = macos_vm_stats() {
            let available = stats.free + stats.inactive;
            return Some(format!(
                "Memory: {} free, {} inactive ({} total available)",
                fmt_gb(stats.free),
                fmt_gb(stats.inactive),
                fmt_gb(available),
            ));
        }
    }
    None
}

/// Format bytes as a human-readable size (e.g. "11.7 GB").
pub fn fmt_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fmt_gb_zero() {
        assert_eq!(fmt_gb(0), "0.0 GB");
    }

    #[test]
    fn fmt_gb_one_gb() {
        assert_eq!(fmt_gb(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn fmt_gb_fractional() {
        assert_eq!(fmt_gb(14_600_000_000), "14.6 GB");
    }

    #[test]
    fn fmt_gb_large() {
        assert_eq!(fmt_gb(128_000_000_000), "128.0 GB");
    }

    #[test]
    fn fmt_gb_small() {
        assert_eq!(fmt_gb(500_000_000), "0.5 GB");
    }

    #[test]
    fn gpu_dtype_always_f32_for_image_gen() {
        // Image generation always uses F32 to avoid BF16 precision issues in denoising
        assert_eq!(gpu_dtype(&Device::Cpu), DType::F32);
    }

    #[test]
    fn gpu_compute_dtype_f32_for_cpu() {
        assert_eq!(gpu_compute_dtype(&Device::Cpu), DType::F32);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn free_system_memory_returns_positive() {
        let mem = free_system_memory_bytes();
        assert!(mem.is_some());
        assert!(mem.unwrap() > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_includes_inactive() {
        let free = free_system_memory_bytes().unwrap();
        let available = available_system_memory_bytes().unwrap();
        assert!(available >= free);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn memory_status_string_is_some() {
        let status = memory_status_string();
        assert!(status.is_some());
        let s = status.unwrap();
        assert!(s.contains("Memory:"));
        assert!(s.contains("free"));
        assert!(s.contains("available"));
    }

    const GB: u64 = 1_000_000_000;

    #[test]
    fn preflight_ok_with_no_memory_info() {
        // On non-macOS or when memory info unavailable, should pass
        let _ = preflight_memory_check("test", 5 * GB);
    }
}
