use anyhow::{Result, bail};

/// Information about decoded VOC audio
pub struct VocInfo {
    pub sample_rate: u32,
    pub pcm_data: Vec<u8>,
}

/// Parse VOC block chain and extract PCM audio data.
/// VOC data inside SBL blocks does NOT have the "Creative Voice File" header —
/// it starts directly with VOC block types.
pub fn parse_voc_blocks(data: &[u8]) -> Result<VocInfo> {
    let mut pos = 0;
    let mut pcm_data = Vec::new();
    let mut sample_rate: u32 = 11025; // default

    // Check if data starts with "Creative Voice File" header (26 bytes)
    if data.len() >= 26 && &data[0..19] == b"Creative Voice File" {
        // Skip the VOC file header
        let header_size = u16::from_le_bytes([data[20], data[21]]) as usize;
        pos = header_size;
    }

    while pos < data.len() {
        let block_type = data[pos];
        pos += 1;

        match block_type {
            0x00 => break, // Terminator
            0x01 => {
                // Sound data block
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3;

                if size < 2 || pos + size > data.len() {
                    break;
                }

                let rate_code = data[pos];
                let _compression = data[pos + 1];
                sample_rate = 1_000_000 / (256 - rate_code as u32);
                pcm_data.extend_from_slice(&data[pos + 2..pos + size]);
                pos += size;
            }
            0x02 => {
                // Sound continuation block
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3;

                if pos + size > data.len() {
                    break;
                }
                pcm_data.extend_from_slice(&data[pos..pos + size]);
                pos += size;
            }
            0x03 => {
                // Silence block
                if pos + 3 > data.len() {
                    break;
                }
                let duration = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                let rate_code = data[pos + 2];
                sample_rate = 1_000_000 / (256 - rate_code as u32);
                pos += 3;
                // Add silence (0x80 = silence for unsigned 8-bit PCM)
                pcm_data.extend(std::iter::repeat(0x80).take(duration + 1));
            }
            0x05 => {
                // Text block - skip
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3 + size;
            }
            0x06 => {
                // Repeat start - skip
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3 + size;
            }
            0x07 => {
                // Repeat end
                if pos + 3 > data.len() {
                    break;
                }
                pos += 3; // 3-byte size (always 0)
            }
            0x09 => {
                // Extended sound data (new format)
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3;

                if size < 12 || pos + size > data.len() {
                    break;
                }
                sample_rate = u32::from_le_bytes([
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                ]);
                let _bits_per_sample = data[pos + 4];
                let _channels = data[pos + 5];
                // Skip 6 bytes of format info, rest is PCM data
                pcm_data.extend_from_slice(&data[pos + 12..pos + size]);
                pos += size;
            }
            _ => {
                // Unknown block type - try to skip with 3-byte size
                if pos + 3 > data.len() {
                    break;
                }
                let size = data[pos] as usize
                    | ((data[pos + 1] as usize) << 8)
                    | ((data[pos + 2] as usize) << 16);
                pos += 3 + size;
            }
        }
    }

    if pcm_data.is_empty() {
        bail!("No PCM audio data found in VOC blocks");
    }

    Ok(VocInfo { sample_rate, pcm_data })
}

/// Convert PCM audio data to a WAV file (unsigned 8-bit mono PCM).
pub fn pcm_to_wav(pcm_data: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_size = pcm_data.len() as u32;
    let file_size = 36 + data_size;
    let channels: u16 = 1;
    let bits_per_sample: u16 = 8;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut wav = Vec::with_capacity(44 + pcm_data.len());

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // PCM format
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm_data);

    wav
}

/// Convert VOC data to WAV. Returns (wav_bytes, sample_rate).
pub fn voc_to_wav(voc_data: &[u8]) -> Result<(Vec<u8>, u32)> {
    let info = parse_voc_blocks(voc_data)?;
    let wav = pcm_to_wav(&info.pcm_data, info.sample_rate);
    Ok((wav, info.sample_rate))
}
