pub mod decode;
pub mod encode;
pub mod mel;
pub mod playback;
pub mod record;
pub mod resample;

pub use decode::{load_bytes, load_file};
pub use encode::{encode_wav, save_wav};
pub use mel::{MelConfig, mel_spectrogram};
pub use playback::{PlaybackHandle, play, play_async};
pub use record::{RecordConfig, record_microphone};
pub use resample::resample;

/// Uniform PCM audio representation used throughout nexo-ai.
///
/// Samples are stored as interleaved f32 when multi-channel.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved PCM samples in f32 format, range [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 44100, 16000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
}

impl AudioBuffer {
    /// Create a new `AudioBuffer` from raw components.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Duration of the audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Number of sample frames (total samples / channels).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Convert to mono by averaging all channels per frame.
    ///
    /// Returns a new mono `AudioBuffer`. If already mono, returns a clone.
    pub fn to_mono(&self) -> Self {
        if self.channels <= 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let mono_samples: Vec<f32> = self
            .samples
            .chunks_exact(ch)
            .map(|frame| {
                let sum: f32 = frame.iter().sum();
                sum / ch as f32
            })
            .collect();

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_buffer() {
        let buf = AudioBuffer::new(vec![0.0; 100], 44100, 2);
        assert_eq!(buf.samples.len(), 100);
        assert_eq!(buf.sample_rate, 44100);
        assert_eq!(buf.channels, 2);
    }

    #[test]
    fn duration_secs_stereo() {
        // 44100 stereo samples = 22050 frames = 0.5s at 44100 Hz
        let buf = AudioBuffer::new(vec![0.0; 44100], 44100, 2);
        let dur = buf.duration_secs();
        assert!((dur - 0.5).abs() < 1e-9, "expected 0.5, got {dur}");
    }

    #[test]
    fn duration_secs_mono() {
        let buf = AudioBuffer::new(vec![0.0; 16000], 16000, 1);
        let dur = buf.duration_secs();
        assert!((dur - 1.0).abs() < 1e-9, "expected 1.0, got {dur}");
    }

    #[test]
    fn duration_secs_zero_rate() {
        let buf = AudioBuffer::new(vec![0.0; 100], 0, 1);
        assert_eq!(buf.duration_secs(), 0.0);
    }

    #[test]
    fn num_frames_stereo() {
        let buf = AudioBuffer::new(vec![0.0; 100], 44100, 2);
        assert_eq!(buf.num_frames(), 50);
    }

    #[test]
    fn num_frames_mono() {
        let buf = AudioBuffer::new(vec![0.0; 100], 44100, 1);
        assert_eq!(buf.num_frames(), 100);
    }

    #[test]
    fn to_mono_already_mono() {
        let buf = AudioBuffer::new(vec![1.0, 2.0, 3.0], 16000, 1);
        let mono = buf.to_mono();
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_mono_stereo() {
        // Stereo: (0.0, 1.0), (0.5, 0.5), (1.0, 0.0)
        let buf = AudioBuffer::new(vec![0.0, 1.0, 0.5, 0.5, 1.0, 0.0], 44100, 2);
        let mono = buf.to_mono();
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 3);
        assert!((mono.samples[0] - 0.5).abs() < 1e-6);
        assert!((mono.samples[1] - 0.5).abs() < 1e-6);
        assert!((mono.samples[2] - 0.5).abs() < 1e-6);
    }
}
