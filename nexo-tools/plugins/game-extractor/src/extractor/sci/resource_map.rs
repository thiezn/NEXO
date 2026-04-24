use super::version::{ResMapVersion, SciGameInfo};
use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResourceType {
    View = 0,
    Pic = 1,
    Script = 2,
    Text = 3,
    Sound = 4,
    Memory = 5,
    Vocab = 6,
    Font = 7,
    Cursor = 8,
    Patch = 9,
    Bitmap = 10,
    Palette = 11,
    CdAudio = 12,
    Audio = 13,
    Sync = 14,
    Message = 15,
    Map = 16,
    Heap = 17,
    Audio36 = 18,
    Sync36 = 19,
}

impl ResourceType {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::View),
            1 => Some(Self::Pic),
            2 => Some(Self::Script),
            3 => Some(Self::Text),
            4 => Some(Self::Sound),
            5 => Some(Self::Memory),
            6 => Some(Self::Vocab),
            7 => Some(Self::Font),
            8 => Some(Self::Cursor),
            9 => Some(Self::Patch),
            10 => Some(Self::Bitmap),
            11 => Some(Self::Palette),
            12 => Some(Self::CdAudio),
            13 => Some(Self::Audio),
            14 => Some(Self::Sync),
            15 => Some(Self::Message),
            16 => Some(Self::Map),
            17 => Some(Self::Heap),
            18 => Some(Self::Audio36),
            19 => Some(Self::Sync36),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::View => "view",
            Self::Pic => "pic",
            Self::Script => "script",
            Self::Text => "text",
            Self::Sound => "sound",
            Self::Memory => "memory",
            Self::Vocab => "vocab",
            Self::Font => "font",
            Self::Cursor => "cursor",
            Self::Patch => "patch",
            Self::Bitmap => "bitmap",
            Self::Palette => "palette",
            Self::CdAudio => "cdaudio",
            Self::Audio => "audio",
            Self::Sync => "sync",
            Self::Message => "message",
            Self::Map => "map",
            Self::Heap => "heap",
            Self::Audio36 => "audio36",
            Self::Sync36 => "sync36",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceEntry {
    pub res_type: ResourceType,
    pub number: u16,
    pub volume: u16,
    pub offset: u32,
}

/// Parse the resource map file and return all resource entries.
pub fn parse_resource_map(game: &SciGameInfo) -> Result<Vec<ResourceEntry>> {
    let data = std::fs::read(&game.map_path).context("Failed to read resource map")?;

    match game.map_version {
        ResMapVersion::Sci0 => parse_map_sci0(&data, 26), // 6-bit volume, 26-bit offset
        ResMapVersion::Sci1Middle => parse_map_sci0(&data, 28), // 4-bit volume, 28-bit offset
        ResMapVersion::Sci1Late => parse_map_sci1(&data, 6),
        ResMapVersion::Sci11 => parse_map_sci1(&data, 5),
        ResMapVersion::Sci2 => parse_map_sci2(&data),
    }
}

/// Parse SCI0/SCI01/SCI1Middle resource map.
/// 6-byte entries: id(2B LE) + offset(4B LE), terminated by offset == 0xFFFFFFFF
/// vol_shift: 26 for SCI0 (6-bit volume), 28 for SCI1Middle (4-bit volume)
fn parse_map_sci0(data: &[u8], vol_shift: u32) -> Result<Vec<ResourceEntry>> {
    let mut entries = Vec::new();
    let mut pos = 0;
    let offset_mask: u32 = (1 << vol_shift) - 1;

    while pos + 6 <= data.len() {
        let id = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let offset_raw =
            u32::from_le_bytes([data[pos + 2], data[pos + 3], data[pos + 4], data[pos + 5]]);

        if offset_raw == 0xFFFFFFFF {
            break;
        }

        let res_type_val = (id >> 11) as u8;
        let number = id & 0x7FF;

        let volume = (offset_raw >> vol_shift) as u16;
        let file_offset = offset_raw & offset_mask;

        if let Some(res_type) = ResourceType::from_u8(res_type_val) {
            entries.push(ResourceEntry {
                res_type,
                number,
                volume,
                offset: file_offset,
            });
        }

        pos += 6;
    }

    Ok(entries)
}

/// Parse SCI1Late/SCI1.1 resource map with type directory headers.
/// Directory: type(1B) + offset(2B LE) per type, terminated by type & 0x1F == 0x1F
/// Entries: number(2B LE) + vol_offset (entry_size - 2 bytes)
fn parse_map_sci1(data: &[u8], entry_size: usize) -> Result<Vec<ResourceEntry>> {
    let mut entries = Vec::new();

    // Parse directory headers
    let mut dir_entries: Vec<(u8, usize)> = Vec::new(); // (type, offset)
    let mut pos = 0;

    while pos + 3 <= data.len() {
        let type_byte = data[pos];
        let dir_offset = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;

        let res_type = type_byte & 0x1F;
        dir_entries.push((type_byte, dir_offset));
        pos += 3;

        if res_type == 0x1F || type_byte == 0xFF {
            break;
        }
    }

    // Parse resource entries per type
    for i in 0..dir_entries.len() {
        let (type_byte, start_offset) = dir_entries[i];
        let res_type_val = type_byte & 0x1F;

        if res_type_val == 0x1F {
            break;
        }

        let end_offset = if i + 1 < dir_entries.len() {
            dir_entries[i + 1].1
        } else {
            data.len()
        };

        let res_type = match ResourceType::from_u8(res_type_val) {
            Some(t) => t,
            None => continue,
        };

        let mut epos = start_offset;
        while epos + entry_size <= end_offset {
            let number = u16::from_le_bytes([data[epos], data[epos + 1]]);

            if entry_size == 6 {
                // SCI1Late: number(2) + vol_offset(4)
                let vol_offset = u32::from_le_bytes([
                    data[epos + 2],
                    data[epos + 3],
                    data[epos + 4],
                    data[epos + 5],
                ]);
                let volume = (vol_offset >> 28) as u16;
                let offset = vol_offset & 0x0FFFFFFF;

                entries.push(ResourceEntry {
                    res_type,
                    number,
                    volume,
                    offset,
                });
            } else {
                // SCI1.1: number(2) + offset(3 bytes LE, shifted left by 1)
                // No volume bits — all resources are in volume 0
                let b0 = data[epos + 2] as u32;
                let b1 = data[epos + 3] as u32;
                let b2 = data[epos + 4] as u32;
                let raw_offset = b0 | (b1 << 8) | (b2 << 16);
                let offset = raw_offset << 1; // word-aligned

                entries.push(ResourceEntry {
                    res_type,
                    number,
                    volume: 0,
                    offset,
                });
            }

            epos += entry_size;
        }
    }

    Ok(entries)
}

/// Parse SCI2/SCI2.1 resource map (RESMAP.000 format).
/// Same directory format as SCI1 but 6-byte entries with no volume bits.
fn parse_map_sci2(data: &[u8]) -> Result<Vec<ResourceEntry>> {
    let mut entries = Vec::new();

    // Parse directory headers
    let mut dir_entries: Vec<(u8, usize)> = Vec::new();
    let mut pos = 0;

    while pos + 3 <= data.len() {
        let type_byte = data[pos];
        let dir_offset = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;

        dir_entries.push((type_byte, dir_offset));
        pos += 3;

        if type_byte == 0xFF || (type_byte & 0x1F) == 0x1F {
            break;
        }
    }

    for i in 0..dir_entries.len() {
        let (type_byte, start_offset) = dir_entries[i];
        if type_byte == 0xFF || (type_byte & 0x1F) == 0x1F {
            break;
        }

        let res_type_val = type_byte & 0x1F;
        let res_type = match ResourceType::from_u8(res_type_val) {
            Some(t) => t,
            None => continue,
        };

        let end_offset = if i + 1 < dir_entries.len() {
            dir_entries[i + 1].1
        } else {
            data.len()
        };

        let mut epos = start_offset;
        while epos + 6 <= end_offset {
            let number = u16::from_le_bytes([data[epos], data[epos + 1]]);
            let offset = u32::from_le_bytes([
                data[epos + 2],
                data[epos + 3],
                data[epos + 4],
                data[epos + 5],
            ]);

            entries.push(ResourceEntry {
                res_type,
                number,
                volume: 0,
                offset,
            });

            epos += 6;
        }
    }

    Ok(entries)
}
