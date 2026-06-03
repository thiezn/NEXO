use rustfft::{FftPlanner, num_complex::Complex};

/// Configuration for mel spectrogram extraction.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Target sample rate in Hz.
    pub sample_rate: u32,
    /// Number of samples per analysis frame.
    pub frame_length: usize,
    /// Hop between successive frames in samples.
    pub hop_length: usize,
    /// FFT size (padded). Must be >= frame_length.
    pub n_fft: usize,
    /// Number of mel filter bands.
    pub n_mels: usize,
    /// Floor value for log-mel output.
    pub mel_floor: f32,
    /// Preemphasis coefficient (0.0 = disabled).
    pub preemphasis: f32,
}

impl MelConfig {
    /// Default configuration for Gemma 4 audio encoder.
    pub fn gemma4() -> Self {
        Self {
            sample_rate: 16000,
            frame_length: 320,
            hop_length: 160,
            n_fft: 512,
            n_mels: 128,
            mel_floor: 1e-3,
            preemphasis: 0.97,
        }
    }

    /// Default configuration for Whisper models.
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16000,
            frame_length: 400,
            hop_length: 160,
            n_fft: 400,
            n_mels: 128,
            mel_floor: 0.0,
            preemphasis: 0.0,
        }
    }
}

/// Convert frequency in Hz to the mel scale (HTK formula).
pub fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Convert mel value back to Hz.
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Compute a mel filterbank matrix as a flat `Vec<f32>` with shape
/// `[n_mels, n_fft / 2 + 1]`.
///
/// Uses the HTK mel scale with Slaney normalization.
pub fn compute_mel_filters(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sample_rate = sample_rate as f32;

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    let f_min = 0.0_f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

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

        // Slaney normalization
        let enorm = 2.0 / (f_right - f_left);
        for j in 0..n_freqs {
            filters[m * n_freqs + j] *= enorm;
        }
    }

    filters
}

/// Extract a mel spectrogram from mono PCM audio.
///
/// Returns a flat `Vec<f32>` of shape `[num_frames, n_mels]`.
/// Uses semicausal padding (prepend `frame_length/2` zeros).
pub fn mel_spectrogram(pcm: &[f32], config: &MelConfig) -> (Vec<f32>, usize) {
    if pcm.is_empty() {
        return (Vec::new(), 0);
    }

    let n_freqs = config.n_fft / 2 + 1;
    let filters = compute_mel_filters(config.n_mels, config.n_fft, config.sample_rate);
    let window = hann_window(config.frame_length);

    // Semicausal padding: prepend frame_length/2 zeros
    let left_pad = config.frame_length / 2;
    let padded_len = left_pad + pcm.len();
    let mut padded = vec![0.0f32; padded_len];
    padded[left_pad..].copy_from_slice(pcm);

    // Compute number of frames
    let num_frames = if padded_len >= config.frame_length {
        (padded_len - config.frame_length) / config.hop_length + 1
    } else {
        0
    };

    if num_frames == 0 {
        return (Vec::new(), 0);
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.n_fft);

    let mut output = Vec::with_capacity(num_frames * config.n_mels);

    for frame_idx in 0..num_frames {
        let start = frame_idx * config.hop_length;
        let end = start + config.frame_length;
        let mut frame: Vec<f32> = padded[start..end].to_vec();

        // Preemphasis
        if config.preemphasis > 0.0 {
            for i in (1..frame.len()).rev() {
                frame[i] -= config.preemphasis * frame[i - 1];
            }
        }

        // Apply Hann window
        for (s, w) in frame.iter_mut().zip(window.iter()) {
            *s *= w;
        }

        // Zero-pad to FFT length
        let mut fft_buf: Vec<Complex<f32>> = frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
        fft_buf.resize(config.n_fft, Complex::new(0.0, 0.0));

        // FFT
        fft.process(&mut fft_buf);

        // Power spectrum (magnitude squared)
        let power: Vec<f32> = fft_buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

        // Apply mel filterbank
        for m in 0..config.n_mels {
            let mut mel_val = 0.0f32;
            for j in 0..n_freqs {
                mel_val += filters[m * n_freqs + j] * power[j];
            }
            // Apply floor and log
            mel_val = mel_val.max(config.mel_floor);
            output.push(mel_val.ln());
        }
    }

    (output, num_frames)
}

/// Compute a Hann window of the given length.
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let phase = std::f32::consts::PI * 2.0 * i as f32 / length as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn mel_filters_shape() {
        let filters = compute_mel_filters(128, 512, 16000);
        let n_freqs = 512 / 2 + 1; // 257
        assert_eq!(filters.len(), 128 * n_freqs);
    }

    #[test]
    fn mel_filters_non_negative() {
        let filters = compute_mel_filters(128, 512, 16000);
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
    fn mel_spectrogram_empty_input() {
        let config = MelConfig::gemma4();
        let (output, frames) = mel_spectrogram(&[], &config);
        assert!(output.is_empty());
        assert_eq!(frames, 0);
    }

    #[test]
    fn mel_spectrogram_short_audio() {
        let config = MelConfig::gemma4();
        // 1 second of silence at 16kHz
        let pcm = vec![0.0f32; 16000];
        let (output, frames) = mel_spectrogram(&pcm, &config);
        assert!(frames > 0);
        assert_eq!(output.len(), frames * config.n_mels);
    }

    #[test]
    fn mel_spectrogram_sine_wave() {
        let config = MelConfig::gemma4();
        // 1 second of 440Hz sine wave at 16kHz
        let pcm: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let (output, frames) = mel_spectrogram(&pcm, &config);
        assert!(frames > 0);
        assert_eq!(output.len(), frames * config.n_mels);
        // Should have non-trivial energy (not all at floor)
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > config.mel_floor.ln());
    }

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(256);
        assert_eq!(w.len(), 256);
        // Hann window starts and ends near zero
        assert!(w[0].abs() < 1e-6);
        // Peak near center
        let mid = w[128];
        assert!(mid > 0.9);
    }

    #[test]
    fn whisper_config_matches_constants() {
        let config = MelConfig::whisper();
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn gemma4_config_defaults() {
        let config = MelConfig::gemma4();
        assert_eq!(config.frame_length, 320);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.n_fft, 512);
        assert_eq!(config.n_mels, 128);
    }
}
