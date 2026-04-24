use crate::extractor::common::audio_convert;
use anyhow::Result;

/// Extracted sound data.
pub struct SoundResource {
    pub number: u16,
    pub format: SoundFormat,
    pub data: Vec<u8>,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum SoundFormat {
    Midi,
    DigitalPcm,
    Raw,
}

impl SoundFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            SoundFormat::Midi => "mid",
            SoundFormat::DigitalPcm => "wav",
            SoundFormat::Raw => "snd",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            SoundFormat::Midi => "MIDI",
            SoundFormat::DigitalPcm => "PCM",
            SoundFormat::Raw => "Raw",
        }
    }
}

/// Extract sound resource data, attempting to identify the format.
/// SCI sound resources contain a header followed by device-specific tracks.
pub fn extract_sound(data: &[u8], number: u16) -> Result<Vec<SoundResource>> {
    let mut results = Vec::new();

    if data.len() < 2 {
        return Ok(results);
    }

    // Check for digital audio header
    if data.len() >= 44 && &data[0..4] == b"RIFF" {
        // WAV file
        results.push(SoundResource {
            number,
            format: SoundFormat::DigitalPcm,
            data: data.to_vec(),
            sample_rate: None,
        });
        return Ok(results);
    }

    // Check for SOL audio (Sierra On-Line audio format)
    if data.len() >= 6 && data[0] == 0x8D {
        // SOL format header
        if let Some(wav) = try_convert_sol_to_wav(data) {
            results.push(SoundResource {
                number,
                format: SoundFormat::DigitalPcm,
                data: wav,
                sample_rate: None,
            });
            return Ok(results);
        }
    }

    // Try to extract MIDI data from SCI sound resource
    if let Some(midi) = try_extract_midi(data) {
        results.push(SoundResource {
            number,
            format: SoundFormat::Midi,
            data: midi,
            sample_rate: None,
        });
    }

    // Also save the raw resource for archival
    if results.is_empty() {
        results.push(SoundResource {
            number,
            format: SoundFormat::Raw,
            data: data.to_vec(),
            sample_rate: None,
        });
    }

    Ok(results)
}

/// Try to extract MIDI data from a SCI sound resource.
/// SCI sound resources have a header with device priorities followed by MIDI tracks.
fn try_extract_midi(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 20 {
        return None;
    }

    // SCI0 sound format: byte 0 = digital sample flag
    // Then pairs of (device_type, offset) terminated by 0xFF
    let mut pos = 0;

    // Skip digital sample flag
    let has_digital = data[0] != 0;
    if has_digital {
        pos += 2; // skip flag + extra byte
    }

    // Find MIDI track data offset
    // Device priority list: each entry is 2 bytes (device, cumulative offset)
    // Terminated by 0xFF
    let mut midi_offset: Option<usize> = None;

    while pos < data.len() {
        if data[pos] == 0xFF {
            pos += 1;
            midi_offset = Some(pos);
            break;
        }
        pos += 2; // skip device entry
    }

    let midi_start = midi_offset?;
    if midi_start >= data.len() {
        return None;
    }

    // Try to build a standard MIDI file
    let mut midi = Vec::new();

    // MIDI header
    midi.extend_from_slice(b"MThd");
    midi.extend_from_slice(&(6u32).to_be_bytes()); // header length
    midi.extend_from_slice(&(0u16).to_be_bytes()); // format 0
    midi.extend_from_slice(&(1u16).to_be_bytes()); // 1 track
    midi.extend_from_slice(&(120u16).to_be_bytes()); // 120 ticks per quarter

    // Track header (placeholder — we'll fill in the length)
    midi.extend_from_slice(b"MTrk");
    let track_len_pos = midi.len();
    midi.extend_from_slice(&[0, 0, 0, 0]); // placeholder

    // Copy MIDI events from SCI data
    let track_data = &data[midi_start..];
    let mut track_pos = 0;

    while track_pos < track_data.len() {
        let byte = track_data[track_pos];

        if byte == 0xFC {
            // End of SCI track
            break;
        }

        midi.push(byte);
        track_pos += 1;
    }

    // Add end-of-track meta event
    midi.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    // Fill in track length
    let track_len = (midi.len() - track_len_pos - 4) as u32;
    midi[track_len_pos..track_len_pos + 4].copy_from_slice(&track_len.to_be_bytes());

    Some(midi)
}

/// Try to convert SOL audio to WAV.
fn try_convert_sol_to_wav(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 12 {
        return None;
    }

    // SOL header: varies by version
    // Try to find PCM data and wrap in WAV
    // This is a simplified implementation — SOL format details vary

    // Check for simple unsigned 8-bit PCM at common sample rates
    let header_size = data[1] as usize;
    if header_size >= data.len() {
        return None;
    }

    let sample_rate: u32 = if data.len() >= 8 {
        u16::from_le_bytes([data[4], data[5]]) as u32
    } else {
        11025
    };

    if sample_rate == 0 || sample_rate > 48000 {
        return None;
    }

    let pcm_data = &data[header_size..];
    if pcm_data.is_empty() {
        return None;
    }

    Some(audio_convert::pcm_to_wav(pcm_data, sample_rate))
}
