use candle_transformers::models::whisper::{N_FFT, SAMPLE_RATE};

/// Compute a mel filterbank matrix as a flat `Vec<f32>` with shape
/// `[n_mels, n_fft / 2 + 1]`.
///
/// Uses the HTK mel scale: `mel(f) = 2595 * log10(1 + f / 700)`.
/// The resulting filters match the format expected by
/// `candle_transformers::models::whisper::audio::pcm_to_mel`.
pub fn compute_mel_filters(n_mels: usize) -> Vec<f32> {
    crate::audio::mel::compute_mel_filters(n_mels, N_FFT, SAMPLE_RATE as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::mel::{hz_to_mel, mel_to_hz};

    #[test]
    fn mel_filter_shape_128() {
        let filters = compute_mel_filters(128);
        let n_freqs = N_FFT / 2 + 1; // 201
        assert_eq!(filters.len(), 128 * n_freqs);
    }

    #[test]
    fn mel_filter_shape_80() {
        let filters = compute_mel_filters(80);
        let n_freqs = N_FFT / 2 + 1;
        assert_eq!(filters.len(), 80 * n_freqs);
    }

    #[test]
    fn mel_filters_non_negative() {
        let filters = compute_mel_filters(128);
        assert!(filters.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn mel_hz_roundtrip() {
        let freq = 1000.0;
        let mel = hz_to_mel(freq);
        let back = mel_to_hz(mel);
        assert!((freq - back).abs() < 0.01);
    }

    #[test]
    fn mel_filters_have_mostly_nonzero_rows() {
        let n_mels = 128;
        let n_freqs = N_FFT / 2 + 1;
        let filters = compute_mel_filters(n_mels);
        let nonzero_rows = (0..n_mels)
            .filter(|&m| {
                let row_sum: f32 = (0..n_freqs).map(|j| filters[m * n_freqs + j]).sum();
                row_sum > 0.0
            })
            .count();
        // The lowest mel bins may be below FFT bin resolution (40 Hz spacing)
        assert!(
            nonzero_rows >= n_mels - 5,
            "only {nonzero_rows}/{n_mels} rows are nonzero"
        );
    }
}
