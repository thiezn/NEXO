use std::path::Path;

fn wav_spec(sample_rate: u32) -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

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

/// Encode PCM f32 samples to WAV bytes in memory (16-bit mono).
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let writer = hound::WavWriter::new(&mut buf, wav_spec(sample_rate))?;
        write_and_finalize(writer, samples)?;
    }
    Ok(buf.into_inner())
}

/// Save PCM f32 samples to a WAV file (16-bit mono).
pub fn save_wav(samples: &[f32], sample_rate: u32, path: &Path) -> anyhow::Result<()> {
    let writer = hound::WavWriter::create(path, wav_spec(sample_rate))?;
    write_and_finalize(writer, samples)
}
