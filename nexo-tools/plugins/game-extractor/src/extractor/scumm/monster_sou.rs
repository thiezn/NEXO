use crate::extractor::common::audio_convert;
use anyhow::{Context, Result, bail};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub struct SpeechEntry {
    pub index: usize,
    pub audio_data: Vec<u8>,
    pub format: AudioFormat,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    Ogg,
    Raw,
}

impl AudioFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Flac => "flac",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Raw => "bin",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Flac => "flac",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Raw => "raw",
        }
    }
}

/// Detect the type of monster sound file from its extension.
fn detect_format(path: &Path) -> AudioFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => match ext.to_lowercase().as_str() {
            "sou" => AudioFormat::Wav, // Will be converted from VOC
            "sof" => AudioFormat::Flac,
            "so3" => AudioFormat::Mp3,
            "sog" => AudioFormat::Ogg,
            _ => AudioFormat::Raw,
        },
        None => AudioFormat::Raw,
    }
}

/// Parse MONSTER.SOU (uncompressed VOC-based speech file).
/// Format: Sequence of "SOU " + size headers, each containing a Creative VOC file.
fn parse_uncompressed_sou(path: &Path) -> Result<Vec<SpeechEntry>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut entries = Vec::new();
    let mut pos: u64 = 0;
    let mut index: usize = 0;

    while pos + 8 < file_len {
        reader.seek(SeekFrom::Start(pos))?;

        let mut header = [0u8; 8];
        if reader.read_exact(&mut header).is_err() {
            break;
        }

        let tag = &header[0..4];

        if tag == b"SOU " {
            let size = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);

            if size <= 8 || pos + size as u64 > file_len {
                pos += 1;
                continue;
            }

            // Read the block data (skip the 8-byte header)
            let data_size = (size - 8) as usize;
            let mut block_data = vec![0u8; data_size];
            if reader.read_exact(&mut block_data).is_err() {
                break;
            }

            // Try to find VOC data within this block
            // Look for "Creative Voice File" header or VCTL/Crea tags
            if let Some(entry) = try_extract_speech_from_block(&block_data, index) {
                entries.push(entry);
            }

            pos += size as u64;
            index += 1;
        } else if tag == b"VCTL" {
            // VCTL block followed by speech data
            let size = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
            // Skip VCTL, next block should be the audio
            pos += size as u64;
            // Don't increment index - the audio block will follow
        } else if tag == b"Crea" || &header[0..3] == b"Cre" {
            // Creative VOC file
            reader.seek(SeekFrom::Start(pos))?;
            // Read until we find the end (look for terminator or next SOU block)
            let remaining = (file_len - pos) as usize;
            let read_size = remaining.min(1024 * 1024); // Cap at 1MB per entry
            let mut voc_data = vec![0u8; read_size];
            let bytes_read = reader.read(&mut voc_data)?;
            voc_data.truncate(bytes_read);

            match audio_convert::voc_to_wav(&voc_data) {
                Ok((wav, sr)) => {
                    entries.push(SpeechEntry {
                        index,
                        audio_data: wav,
                        format: AudioFormat::Wav,
                        sample_rate: Some(sr),
                    });
                }
                Err(_) => {}
            }

            // Advance past this block
            pos += read_size as u64;
            index += 1;
        } else {
            pos += 1;
        }
    }

    Ok(entries)
}

fn try_extract_speech_from_block(block_data: &[u8], index: usize) -> Option<SpeechEntry> {
    // Check if it starts with "Creative Voice File"
    if block_data.len() >= 26 && &block_data[0..19] == b"Creative Voice File" {
        match audio_convert::voc_to_wav(block_data) {
            Ok((wav, sr)) => {
                return Some(SpeechEntry {
                    index,
                    audio_data: wav,
                    format: AudioFormat::Wav,
                    sample_rate: Some(sr),
                });
            }
            Err(_) => {}
        }
    }

    // Check for nested blocks
    let mut pos = 0;
    while pos + 8 < block_data.len() {
        let tag = &block_data[pos..pos + 4];
        if tag == b"Crea"
            || (block_data.len() > pos + 26 && &block_data[pos..pos + 19] == b"Creative Voice File")
        {
            match audio_convert::voc_to_wav(&block_data[pos..]) {
                Ok((wav, sr)) => {
                    return Some(SpeechEntry {
                        index,
                        audio_data: wav,
                        format: AudioFormat::Wav,
                        sample_rate: Some(sr),
                    });
                }
                Err(_) => {}
            }
        }

        // Try looking for VOC block type 0x01 directly
        if tag == b"VTLK" || tag == b"TALK" {
            let size = u32::from_be_bytes([
                block_data[pos + 4],
                block_data[pos + 5],
                block_data[pos + 6],
                block_data[pos + 7],
            ]);
            let audio_start = pos + 8;
            let audio_end = (pos + size as usize).min(block_data.len());
            if audio_start < audio_end {
                match audio_convert::voc_to_wav(&block_data[audio_start..audio_end]) {
                    Ok((wav, sr)) => {
                        return Some(SpeechEntry {
                            index,
                            audio_data: wav,
                            format: AudioFormat::Wav,
                            sample_rate: Some(sr),
                        });
                    }
                    Err(_) => {}
                }
            }
        }

        pos += 1;
    }

    None
}

/// Parse compressed MONSTER.SOF/SO3/SOG files.
/// These have a mapping table at the start followed by compressed audio data.
fn parse_compressed_sou(path: &Path, format: AudioFormat) -> Result<Vec<SpeechEntry>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    // Read index table size
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let table_size = u32::from_be_bytes(buf4) as usize;

    if table_size == 0 || table_size > file_len as usize {
        bail!("Invalid mapping table size: {}", table_size);
    }

    // Read mapping table entries (16 bytes each)
    let num_entries = table_size / 16;
    let mut entries_info = Vec::with_capacity(num_entries);

    for _ in 0..num_entries {
        let mut entry = [0u8; 16];
        reader.read_exact(&mut entry)?;

        let _orig_offset = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let new_offset = u32::from_be_bytes([entry[4], entry[5], entry[6], entry[7]]);
        let num_tags = u32::from_be_bytes([entry[8], entry[9], entry[10], entry[11]]);
        let compressed_size = u32::from_be_bytes([entry[12], entry[13], entry[14], entry[15]]);

        entries_info.push((new_offset, num_tags, compressed_size));
    }

    let mut results = Vec::new();

    for (idx, &(offset, num_tags, size)) in entries_info.iter().enumerate() {
        if offset as u64 >= file_len || size == 0 {
            continue;
        }

        reader.seek(SeekFrom::Start(4 + offset as u64))?;

        // Skip lip-sync tags (num_tags * 2 bytes)
        let tags_size = num_tags as usize * 2;
        let mut _tags = vec![0u8; tags_size];
        if reader.read_exact(&mut _tags).is_err() {
            continue;
        }

        // Read compressed audio data
        let audio_size = size as usize;
        let mut audio_data = vec![0u8; audio_size];
        if reader.read_exact(&mut audio_data).is_err() {
            continue;
        }

        results.push(SpeechEntry {
            index: idx,
            audio_data,
            format,
            sample_rate: None,
        });
    }

    Ok(results)
}

/// Main entry point: parse a MONSTER.SOU/.sof/.so3/.sog file.
pub fn parse_monster_file(path: &Path) -> Result<Vec<SpeechEntry>> {
    let format = detect_format(path);

    match format {
        AudioFormat::Wav => parse_uncompressed_sou(path),
        AudioFormat::Flac | AudioFormat::Mp3 | AudioFormat::Ogg => {
            parse_compressed_sou(path, format)
        }
        _ => bail!("Unknown monster sound file format: {}", path.display()),
    }
}
