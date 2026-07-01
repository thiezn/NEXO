use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::AudioBuffer;

/// Resample an `AudioBuffer` to a different sample rate.
///
/// Takes ownership of the buffer to avoid an unnecessary clone when the buffer
/// is already at the target rate. Multi-channel audio is handled by processing
/// each channel independently through rubato.
pub fn resample(buffer: AudioBuffer, target_sample_rate: u32) -> anyhow::Result<AudioBuffer> {
    if buffer.sample_rate == target_sample_rate {
        return Ok(buffer);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = target_sample_rate as f64 / buffer.sample_rate as f64;
    let channels = buffer.channels as usize;
    let num_frames = buffer.num_frames();

    if num_frames == 0 {
        return Ok(AudioBuffer::new(
            Vec::new(),
            target_sample_rate,
            buffer.channels,
        ));
    }

    // De-interleave into per-channel vectors
    let mut channel_data: Vec<Vec<f32>> = vec![Vec::with_capacity(num_frames); channels];
    for (i, sample) in buffer.samples.iter().enumerate() {
        channel_data[i % channels].push(*sample);
    }

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, num_frames, channels)?;

    let output = resampler.process(&channel_data, None)?;

    // Re-interleave
    let out_frames = output.first().map_or(0, |ch| ch.len());
    let mut interleaved = Vec::with_capacity(out_frames * channels);
    for frame_idx in 0..out_frames {
        for ch in &output {
            interleaved.push(ch[frame_idx]);
        }
    }

    tracing::info!(
        from = buffer.sample_rate,
        to = target_sample_rate,
        in_frames = num_frames,
        out_frames,
        channels,
        "resampled audio"
    );

    Ok(AudioBuffer::new(
        interleaved,
        target_sample_rate,
        buffer.channels,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate_returns_unchanged() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let buf = AudioBuffer::new(samples.clone(), 16000, 1);
        let result = resample(buf, 16000);
        assert!(result.is_ok());
        let result = match result {
            Ok(r) => r,
            Err(_) => return,
        };
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.samples, samples);
    }

    #[test]
    fn resample_empty_buffer() {
        let buf = AudioBuffer::new(Vec::new(), 44100, 1);
        let result = resample(buf, 16000);
        assert!(result.is_ok());
        let result = match result {
            Ok(r) => r,
            Err(_) => return,
        };
        assert_eq!(result.sample_rate, 16000);
        assert!(result.samples.is_empty());
    }
}
