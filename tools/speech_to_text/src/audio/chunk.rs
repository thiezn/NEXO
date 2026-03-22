use candle_transformers::models::whisper::{N_SAMPLES, SAMPLE_RATE};

/// Splits long audio into 30-second chunks for Whisper processing.
///
/// Supports timestamp-based seeking: after decoding a chunk, call `seek_to_time`
/// with the last decoded timestamp to avoid cutting words mid-syllable.
pub struct AudioChunker {
    samples: Vec<f32>,
    position: usize,
    buffer: Vec<f32>,
}

impl AudioChunker {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            position: 0,
            buffer: vec![0.0f32; N_SAMPLES],
        }
    }

    /// Get the next chunk and its offset in seconds.
    /// Returns a slice into an internal buffer (zero-padded to N_SAMPLES).
    /// Returns None when all audio has been consumed.
    pub fn next_chunk(&mut self) -> Option<(&[f32], f64)> {
        if self.position >= self.samples.len() {
            return None;
        }

        let offset_secs = self.position as f64 / SAMPLE_RATE as f64;
        let remaining = self.samples.len() - self.position;
        let take = remaining.min(N_SAMPLES);

        self.buffer[..take].copy_from_slice(&self.samples[self.position..self.position + take]);
        self.buffer[take..].fill(0.0);

        // Advance by what we actually consumed (not full N_SAMPLES)
        self.position += take;

        Some((&self.buffer, offset_secs))
    }

    /// Seek to a time position in seconds. Only moves forward to prevent infinite loops.
    pub fn seek_to_time(&mut self, seconds: f64) {
        let pos = (seconds * SAMPLE_RATE as f64) as usize;
        let pos = pos.min(self.samples.len());
        if pos > self.position {
            self.position = pos;
        }
    }

    /// Total duration in seconds.
    pub fn total_duration(&self) -> f64 {
        self.samples.len() as f64 / SAMPLE_RATE as f64
    }
}
