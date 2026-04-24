use std::path::Path;

use anyhow::Context;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::AudioBuffer;

/// Load an audio file from disk and decode to PCM f32.
///
/// Preserves the original channel layout. Use [`AudioBuffer::to_mono()`] to
/// down-mix after loading if needed.
pub fn load_file(path: &Path) -> anyhow::Result<AudioBuffer> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    match decode_stream(mss, hint) {
        Ok(audio) => Ok(audio),
        Err(symphonia_error) if is_wav_path(path) => {
            tracing::warn!(
                path = %path.display(),
                error = %symphonia_error,
                "symphonia failed to decode WAV file, falling back to hound"
            );
            decode_wav_reader(std::fs::File::open(path)?).with_context(|| {
                format!(
                    "failed to decode WAV file '{}': {symphonia_error}",
                    path.display()
                )
            })
        }
        Err(error) => Err(error),
    }
}

/// Load audio from in-memory bytes. The format is auto-detected by symphonia.
pub fn load_bytes(data: &[u8]) -> anyhow::Result<AudioBuffer> {
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();

    match decode_stream(mss, hint) {
        Ok(audio) => Ok(audio),
        Err(symphonia_error) if looks_like_wav(data) => {
            tracing::warn!(
                error = %symphonia_error,
                "symphonia failed to decode WAV bytes, falling back to hound"
            );
            decode_wav_reader(std::io::Cursor::new(data))
                .with_context(|| format!("failed to decode WAV bytes: {symphonia_error}"))
        }
        Err(error) => Err(error),
    }
}

fn looks_like_wav(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE"
}

fn is_wav_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
}

fn decode_wav_reader<R>(reader: R) -> anyhow::Result<AudioBuffer>
where
    R: std::io::Read + std::io::Seek,
{
    let mut reader = hound::WavReader::new(reader)?;
    let spec = reader.spec();

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = pcm_scale(spec.bits_per_sample);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    tracing::info!(
        sample_rate = spec.sample_rate,
        channels = spec.channels,
        samples = samples.len(),
        "decoded wav via hound"
    );

    Ok(AudioBuffer {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

fn pcm_scale(bits_per_sample: u16) -> f32 {
    match bits_per_sample {
        0 => 1.0,
        bits => (1_i64 << (bits - 1)) as f32,
    }
}

/// Shared decode logic: probe the stream, find the default audio track,
/// and decode all packets into an `AudioBuffer`.
fn decode_stream(mss: MediaSourceStream, hint: Hint) -> anyhow::Result<AudioBuffer> {
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("no audio track found"))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let num_frames = decoded.capacity();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    tracing::info!(
        sample_rate,
        channels,
        samples = all_samples.len(),
        "decoded audio"
    );

    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels: channels as u16,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn decode_wav_reader_handles_pcm16() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            writer.write_sample(0_i16).unwrap();
            writer.write_sample(i16::MAX).unwrap();
            writer.write_sample(i16::MIN + 1).unwrap();
            writer.finalize().unwrap();
        }

        let audio = decode_wav_reader(std::io::Cursor::new(bytes)).unwrap();

        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 3);
        assert!(audio.samples[1] > 0.99);
        assert!(audio.samples[2] < -0.99);
    }

    #[test]
    fn looks_like_wav_checks_riff_header() {
        assert!(looks_like_wav(b"RIFF\0\0\0\0WAVEfmt "));
        assert!(!looks_like_wav(b"not a wav"));
    }
}
