use super::decode::AudioData;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Resample audio to 16kHz. Takes ownership to avoid cloning when already at 16kHz.
pub fn resample_to_16khz(audio: AudioData) -> anyhow::Result<Vec<f32>> {
    if audio.sample_rate == WHISPER_SAMPLE_RATE {
        return Ok(audio.samples);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = WHISPER_SAMPLE_RATE as f64 / audio.sample_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        audio.samples.len(),
        1, // mono
    )?;

    let input = vec![audio.samples];
    let output = resampler.process(&input, None)?;

    let resampled = output.into_iter().next().unwrap_or_default();
    tracing::info!(
        from = audio.sample_rate,
        to = WHISPER_SAMPLE_RATE,
        out_samples = resampled.len(),
        "resampled audio"
    );

    Ok(resampled)
}
