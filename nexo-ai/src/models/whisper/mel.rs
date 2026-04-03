use candle_transformers::models::whisper::{N_FFT, SAMPLE_RATE};

/// Compute a mel filterbank matrix as a flat `Vec<f32>` with shape
/// `[n_mels, n_fft / 2 + 1]`.
///
/// Uses the HTK mel scale: `mel(f) = 2595 * log10(1 + f / 700)`.
/// The resulting filters match the format expected by
/// `candle_transformers::models::whisper::audio::pcm_to_mel`.
pub fn compute_mel_filters(n_mels: usize) -> Vec<f32> {
    let n_fft = N_FFT;
    let sample_rate = SAMPLE_RATE as f32;
    let n_freqs = n_fft / 2 + 1; // 201

    // Frequency of each FFT bin
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // n_mels + 2 evenly spaced points on the mel scale
    let f_min = 0.0_f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // Build triangular filters
    let mut filters = vec![0.0_f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_left && freq <= f_center && f_center > f_left {
                filters[m * n_freqs + j] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                filters[m * n_freqs + j] = (f_right - freq) / (f_right - f_center);
            }
        }

        // Normalize by the width of the triangle (slaney normalization)
        let enorm = 2.0 / (f_right - f_left);
        for j in 0..n_freqs {
            filters[m * n_freqs + j] *= enorm;
        }
    }

    filters
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
