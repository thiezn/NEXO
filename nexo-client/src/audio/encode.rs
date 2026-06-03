use std::path::Path;

use super::AudioBuffer;

/// Build a `hound::WavSpec` matching the given buffer's layout.
fn wav_spec(buffer: &AudioBuffer) -> hound::WavSpec {
    hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

/// Write all samples to a `hound::WavWriter`, clamping to [-1.0, 1.0] and
/// scaling to 16-bit signed integers.
fn write_and_finalize<W: std::io::Write + std::io::Seek>(
    mut writer: hound::WavWriter<W>,
    samples: &[f32],
) -> anyhow::Result<()> {
    for &sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(scaled)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Save an `AudioBuffer` to a WAV file at the given path (16-bit PCM).
pub fn save_wav(buffer: &AudioBuffer, path: &Path) -> anyhow::Result<()> {
    let writer = hound::WavWriter::create(path, wav_spec(buffer))?;
    write_and_finalize(writer, &buffer.samples)
}

/// Encode an `AudioBuffer` to WAV bytes in memory (16-bit PCM).
pub fn encode_wav(buffer: &AudioBuffer) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let writer = hound::WavWriter::new(&mut cursor, wav_spec(buffer))?;
        write_and_finalize(writer, &buffer.samples)?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::decode;

    #[test]
    fn encode_decode_roundtrip() {
        // Create a simple mono buffer with a known waveform
        let sample_rate = 16000;
        let samples: Vec<f32> = (0..1600)
            .map(|i| (i as f32 / 1600.0 * std::f32::consts::TAU).sin() * 0.8)
            .collect();
        let original = AudioBuffer::new(samples, sample_rate, 1);

        // Encode to WAV bytes
        let wav_bytes = encode_wav(&original).ok();
        assert!(wav_bytes.is_some(), "encode_wav should succeed");
        let wav_bytes = wav_bytes.unwrap_or_default();
        assert!(!wav_bytes.is_empty());

        // Decode back
        let decoded = decode::load_bytes(&wav_bytes);
        assert!(decoded.is_ok(), "load_bytes should succeed on WAV data");
        let decoded = match decoded {
            Ok(d) => d,
            Err(_) => return,
        };

        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), original.samples.len());

        // Verify samples are approximately equal (16-bit quantization introduces error)
        for (i, (orig, dec)) in original
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .enumerate()
        {
            let diff = (orig - dec).abs();
            assert!(
                diff < 0.001,
                "sample {i} differs too much: original={orig}, decoded={dec}, diff={diff}"
            );
        }
    }

    #[test]
    fn encode_stereo() {
        let buffer = AudioBuffer::new(vec![0.5, -0.5, 0.3, -0.3], 44100, 2);
        let wav_bytes = encode_wav(&buffer);
        assert!(wav_bytes.is_ok());
        let bytes = wav_bytes.unwrap_or_default();
        assert!(!bytes.is_empty());
    }
}
