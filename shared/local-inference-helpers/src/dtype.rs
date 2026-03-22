use candle_core::{DType, Device};

/// Select the optimal dtype for GPU inference.
///
/// - CUDA: BF16 (well-supported by tensor cores, standard for diffusion)
/// - Metal/MPS: F32 (BF16 on Metal has precision issues that cause washed-out,
///   blurry images — matmul accumulation errors compound through denoising loops.
///   This matches InvokeAI/diffusers which also avoid BF16 on MPS.)
/// - CPU: F32
pub fn gpu_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_returns_f32() {
        assert_eq!(gpu_dtype(&Device::Cpu), DType::F32);
    }
}
