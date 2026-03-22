use candle_core::Device;

/// Create a GPU device, falling back to CPU if no accelerator is available.
///
/// The `on_info` callback receives informational messages about device selection
/// (e.g. "Metal detected, using GPU"). Pass `|_| {}` to suppress.
///
/// Override with env var `LOCAL_INFERENCE_DEVICE=cpu` to force CPU inference.
pub fn create_device(on_info: impl Fn(&str)) -> anyhow::Result<Device> {
    let force_cpu = std::env::var("LOCAL_INFERENCE_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);
    if force_cpu {
        on_info("CPU forced via LOCAL_INFERENCE_DEVICE=cpu");
        tracing::info!("CPU forced via LOCAL_INFERENCE_DEVICE=cpu");
        return Ok(Device::Cpu);
    }
    if candle_core::utils::cuda_is_available() {
        on_info("CUDA detected, using GPU");
        tracing::info!("CUDA detected, using GPU");
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        on_info("Metal detected, using GPU");
        tracing::info!("Metal detected, using GPU");
        Ok(Device::new_metal(0)?)
    } else {
        on_info("No GPU detected, using CPU");
        tracing::warn!("No GPU detected, falling back to CPU");
        Ok(Device::Cpu)
    }
}

// ── Memory query ────────────────────────────────────────────────────────────

/// Raw VM statistics from macOS host_statistics64.
#[cfg(target_os = "macos")]
struct MacOSMemInfo {
    free: u64,
    inactive: u64,
}

/// Query macOS VM statistics using host_statistics64 FFI.
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

/// Immediately free system memory on macOS (free pages only).
///
/// Conservative metric — memory available WITHOUT reclaiming inactive pages.
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

/// Query free VRAM in bytes from the current CUDA context.
#[cfg(feature = "cuda")]
pub fn free_vram_bytes() -> Option<u64> {
    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

/// On macOS, return immediately free system memory (conservative estimate).
/// On other non-CUDA platforms, no VRAM info available.
#[cfg(not(feature = "cuda"))]
pub fn free_vram_bytes() -> Option<u64> {
    free_system_memory_bytes()
}

// ── Decision functions ──────────────────────────────────────────────────────

/// Determine whether a component should be placed on GPU given free VRAM.
///
/// On Metal (Apple Silicon), always returns true — unified memory means GPU
/// placement is purely a compute performance decision, not a memory one.
/// On CUDA, checks that free discrete VRAM exceeds the threshold.
pub fn should_use_gpu(is_cuda: bool, is_metal: bool, free_vram: u64, threshold: u64) -> bool {
    if is_metal {
        return true;
    }
    is_cuda && free_vram > threshold
}

/// Check whether a model component fits comfortably in memory.
///
/// On CUDA, checks discrete VRAM. On Metal, checks free system memory against threshold.
/// Returns false if loading would require heavy page reclamation.
pub fn fits_in_memory(is_cuda: bool, is_metal: bool, free_vram: u64, threshold: u64) -> bool {
    if is_metal {
        if free_vram > 0 {
            return free_vram > threshold;
        }
        return true;
    }
    is_cuda && free_vram > threshold
}

/// Pre-flight memory guard: check if loading a component of `size_bytes` would
/// exceed available system memory. Hard-fails if >90% of available; warns if >2x free.
pub fn preflight_memory_check(component_name: &str, size_bytes: u64) -> anyhow::Result<()> {
    let available = match available_system_memory_bytes() {
        Some(a) if a > 0 => a,
        _ => return Ok(()),
    };
    let free = free_system_memory_bytes();

    if size_bytes > available * 90 / 100 {
        anyhow::bail!(
            "Not enough memory to load {} ({} needed, {} available).\n\
             Close other applications or use a smaller quantized model.",
            component_name,
            fmt_gb(size_bytes),
            fmt_gb(available),
        );
    }

    if let Some(f) = free
        && size_bytes > f * 2
    {
        tracing::warn!(
            "{} ({}) exceeds free memory ({}), will reclaim inactive pages",
            component_name,
            fmt_gb(size_bytes),
            fmt_gb(f),
        );
    }

    Ok(())
}

/// Format bytes as a human-readable size (e.g. "11.7 GB").
pub fn fmt_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}

/// Return a human-readable memory status string for display.
pub fn memory_status_string() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let Some(free) = free_vram_bytes() {
            return Some(format!("VRAM: {} free", fmt_gb(free)));
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(stats) = macos_vm_stats() {
            let available = stats.free + stats.inactive;
            return Some(format!(
                "Memory: {} free, {} available",
                fmt_gb(stats.free),
                fmt_gb(available),
            ));
        }
    }
    None
}

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

    #[test]
    fn metal_always_uses_gpu() {
        assert!(should_use_gpu(false, true, 0, 16_000_000_000));
    }

    #[test]
    fn metal_fits_when_enough_free() {
        assert!(fits_in_memory(false, true, 20_000_000_000, 16_000_000_000));
    }

    #[test]
    fn metal_does_not_fit_when_free_low() {
        assert!(!fits_in_memory(false, true, 2_000_000_000, 16_000_000_000));
    }

    #[test]
    fn metal_fits_fallback_when_no_memory_info() {
        assert!(fits_in_memory(false, true, 0, 16_000_000_000));
    }

    #[test]
    fn cuda_uses_gpu_with_enough_vram() {
        assert!(should_use_gpu(true, false, 16_700_000_000, 16_000_000_000));
    }

    #[test]
    fn cuda_skips_gpu_with_low_vram() {
        assert!(!should_use_gpu(true, false, 14_600_000_000, 16_000_000_000));
    }

    #[test]
    fn no_gpu_never_uses_gpu() {
        assert!(!should_use_gpu(false, false, 100_000_000_000, 16_000_000_000));
    }

    const GB: u64 = 1_000_000_000;

    #[test]
    fn preflight_ok_with_no_memory_info() {
        // On non-macOS CI or when memory info unavailable, should pass
        // This test just verifies the function doesn't panic
        let _ = preflight_memory_check("test", 5 * GB);
    }
}
