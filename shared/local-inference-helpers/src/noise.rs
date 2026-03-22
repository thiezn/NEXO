use candle_core::{DType, Device, Tensor};

/// Generate a random seed from the current system time.
pub fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Generate deterministic noise on a device with a given seed.
///
/// This is the ONLY correct way to generate initial noise for denoising.
/// All pipelines MUST use this instead of calling `device.set_seed()` +
/// `Tensor::randn()` separately.
///
/// Noise is generated on CPU using a deterministic Rust RNG, then moved to
/// the target device. This guarantees:
/// 1. Same seed always produces identical noise (deterministic)
/// 2. Same seed produces the same noise across CUDA, Metal, and CPU backends
///    (cross-platform reproducibility)
///
/// GPU-native RNG (Metal's HybridTaus, CUDA's cuRAND) use different algorithms
/// that produce different sequences from the same seed. CPU generation avoids this.
pub fn seeded_randn(
    seed: u64,
    shape: &[usize],
    device: &Device,
    dtype: DType,
) -> anyhow::Result<Tensor> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let elem_count: usize = shape.iter().product();
    let noise: Vec<f32> = (0..elem_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    let tensor = Tensor::from_vec(noise, shape, &Device::Cpu)?;
    Ok(tensor.to_dtype(dtype)?.to_device(device)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn produces_correct_shape() {
        let t = seeded_randn(42, &[1, 4, 8, 8], &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn respects_dtype() {
        let t = seeded_randn(42, &[2, 2], &Device::Cpu, DType::BF16).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
    }

    #[test]
    fn deterministic_same_seed() {
        let a = seeded_randn(1337, &[1, 16, 8, 8], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(1337, &[1, 16, 8, 8], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "same seed must produce identical noise");
    }

    #[test]
    fn different_seeds_differ() {
        let a = seeded_randn(42, &[1, 4, 8, 8], &Device::Cpu, DType::F32).unwrap();
        let b = seeded_randn(43, &[1, 4, 8, 8], &Device::Cpu, DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0, "different seeds must produce different noise");
    }
}
