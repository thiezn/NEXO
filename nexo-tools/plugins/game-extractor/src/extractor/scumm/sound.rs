use super::block::{self, Block};
use super::version::ScummVersion;
use crate::extractor::common::audio_convert;

#[derive(Debug, Clone, Copy)]
pub enum SoundType {
    Digital,     // SBL - Sound Blaster / VOC
    Adlib,       // ADL - AdLib MIDI
    Roland,      // ROL - Roland MT-32
    GeneralMidi, // GMD - General MIDI
    PcSpeaker,   // SPK - PC Speaker
    Amiga,       // AMI - Amiga
}

impl SoundType {
    pub fn label(&self) -> &'static str {
        match self {
            SoundType::Digital => "digital",
            SoundType::Adlib => "adlib",
            SoundType::Roland => "roland",
            SoundType::GeneralMidi => "general_midi",
            SoundType::PcSpeaker => "pc_speaker",
            SoundType::Amiga => "amiga",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            SoundType::Digital => "wav",
            SoundType::Adlib | SoundType::Roland | SoundType::GeneralMidi => "mid",
            SoundType::PcSpeaker => "bin",
            SoundType::Amiga => "mid",
        }
    }
}

pub struct ExtractedSound {
    pub sound_type: SoundType,
    pub data: Vec<u8>,
    pub sample_rate: Option<u32>,
}

pub struct RoomSounds {
    pub room_id: u16,
    pub sounds: Vec<(u16, Vec<ExtractedSound>)>, // (sound_index, extracted sounds)
}

/// Extract all sounds from a LFLF block.
/// Sounds are SOUN blocks that are siblings of ROOM inside LFLF.
pub fn extract_room_sounds(
    data: &[u8],
    lflf_block: &Block,
    room_id: u16,
    _version: ScummVersion,
) -> RoomSounds {
    let mut result = RoomSounds {
        room_id,
        sounds: Vec::new(),
    };

    let children = block::iter_children(data, lflf_block);
    let mut sound_idx: u16 = 0;

    for child in &children {
        if &child.tag == b"SOUN" {
            let extracted = extract_soun_block(data, child);
            if !extracted.is_empty() {
                result.sounds.push((sound_idx, extracted));
            }
            sound_idx += 1;
        }
    }

    result
}

/// Parse a SOUN block and extract all sub-sounds.
fn extract_soun_block(data: &[u8], soun: &Block) -> Vec<ExtractedSound> {
    let mut results = Vec::new();

    // SOUN contains a "SOU " wrapper block, or directly contains sub-blocks
    let children = block::iter_children(data, soun);

    for child in &children {
        let tag = &child.tag;
        match tag {
            b"SOU " => {
                // Wrapper - recurse into it
                let inner = extract_sou_wrapper(data, &child);
                results.extend(inner);
            }
            b"SBL " => {
                if let Some(s) = extract_sbl(data, &child) {
                    results.push(s);
                }
            }
            b"ADL " => {
                results.push(extract_midi_block(data, &child, SoundType::Adlib));
            }
            b"ROL " => {
                results.push(extract_midi_block(data, &child, SoundType::Roland));
            }
            b"GMD " => {
                results.push(extract_midi_block(data, &child, SoundType::GeneralMidi));
            }
            b"SPK " => {
                results.push(extract_raw_block(data, &child, SoundType::PcSpeaker));
            }
            b"AMI " => {
                results.push(extract_midi_block(data, &child, SoundType::Amiga));
            }
            _ => {
                // Check if this might be a MIDI block
                if tag == b"MIDI" {
                    results.push(extract_midi_block(data, &child, SoundType::GeneralMidi));
                }
            }
        }
    }

    // If no children found, the SOUN block might directly contain sound data
    if results.is_empty() && soun.size > 8 {
        let soun_data = &data[soun.data_offset()..soun.end_offset()];
        if soun_data.len() >= 8 {
            let inner_tag = &soun_data[0..4];
            if inner_tag == b"SOU " {
                // Parse as SOU wrapper with block header
                if let Ok(inner_block) = block::parse_block(soun_data, 0) {
                    let offset_in_data = soun.data_offset();
                    let adjusted = Block {
                        tag: inner_block.tag,
                        size: inner_block.size,
                        offset: offset_in_data,
                    };
                    results.extend(extract_sou_wrapper(data, &adjusted));
                }
            } else if inner_tag == b"Crea" {
                // Raw Creative Voice File (VOC) data — used by V7 (Full Throttle)
                match audio_convert::voc_to_wav(soun_data) {
                    Ok((wav, sample_rate)) => {
                        results.push(ExtractedSound {
                            sound_type: SoundType::Digital,
                            data: wav,
                            sample_rate: Some(sample_rate),
                        });
                    }
                    Err(_) => {}
                }
            }
        }
    }

    results
}

fn extract_sou_wrapper(data: &[u8], sou: &Block) -> Vec<ExtractedSound> {
    let mut results = Vec::new();
    let children = block::iter_children(data, sou);

    for child in &children {
        match &child.tag {
            b"SBL " => {
                if let Some(s) = extract_sbl(data, &child) {
                    results.push(s);
                }
            }
            b"ADL " => {
                results.push(extract_midi_block(data, &child, SoundType::Adlib));
            }
            b"ROL " => {
                results.push(extract_midi_block(data, &child, SoundType::Roland));
            }
            b"GMD " => {
                results.push(extract_midi_block(data, &child, SoundType::GeneralMidi));
            }
            b"SPK " => {
                results.push(extract_raw_block(data, &child, SoundType::PcSpeaker));
            }
            b"AMI " => {
                results.push(extract_midi_block(data, &child, SoundType::Amiga));
            }
            _ => {}
        }
    }

    results
}

/// Extract SBL block: contains AUhd/AUdt or WVhd/WVdt with VOC audio data.
fn extract_sbl(data: &[u8], sbl: &Block) -> Option<ExtractedSound> {
    let children = block::iter_children(data, sbl);

    // Look for AUdt or WVdt block (the actual audio data)
    for child in &children {
        if &child.tag == b"AUdt" || &child.tag == b"WVdt" {
            let voc_data = &data[child.data_offset()..child.end_offset()];
            if voc_data.is_empty() {
                continue;
            }

            match audio_convert::voc_to_wav(voc_data) {
                Ok((wav, sample_rate)) => {
                    return Some(ExtractedSound {
                        sound_type: SoundType::Digital,
                        data: wav,
                        sample_rate: Some(sample_rate),
                    });
                }
                Err(_) => {
                    // Fall back to raw data
                    return Some(ExtractedSound {
                        sound_type: SoundType::Digital,
                        data: voc_data.to_vec(),
                        sample_rate: None,
                    });
                }
            }
        }
    }

    None
}

fn extract_midi_block(data: &[u8], block: &Block, sound_type: SoundType) -> ExtractedSound {
    let midi_data = data[block.data_offset()..block.end_offset()].to_vec();
    ExtractedSound {
        sound_type,
        data: midi_data,
        sample_rate: None,
    }
}

fn extract_raw_block(data: &[u8], block: &Block, sound_type: SoundType) -> ExtractedSound {
    let raw_data = data[block.data_offset()..block.end_offset()].to_vec();
    ExtractedSound {
        sound_type,
        data: raw_data,
        sample_rate: None,
    }
}
