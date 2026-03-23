use std::collections::HashMap;
use anyhow::{Result, Context};
use super::block::{Block, parse_block};

#[derive(Debug)]
pub struct RoomDirEntry {
    pub room_id: usize,
    pub file_num: u8,
    pub offset: u32,
}

#[derive(Debug)]
pub struct SoundDirEntry {
    pub sound_id: usize,
    pub file_num: u8,
    pub offset: u32,
}

#[derive(Debug)]
pub struct GameIndex {
    pub room_directory: Vec<RoomDirEntry>,
    pub sound_directory: Vec<SoundDirEntry>,
    pub room_names: HashMap<u8, String>,
}

/// Parse the index file (.000) to extract the room and sound directories.
pub fn parse_index(data: &[u8]) -> Result<GameIndex> {
    let mut room_directory = Vec::new();
    let mut sound_directory = Vec::new();
    let mut room_names = HashMap::new();
    let mut pos = 0;

    while pos + 8 <= data.len() {
        let block = match parse_block(data, pos) {
            Ok(b) => b,
            Err(_) => break,
        };

        match &block.tag {
            b"RNAM" => {
                room_names = parse_rnam(data, &block);
            }
            b"DROO" => {
                room_directory = parse_directory(data, &block)
                    .with_context(|| "parsing DROO")?
                    .into_iter()
                    .map(|(id, file_num, offset)| RoomDirEntry { room_id: id, file_num, offset })
                    .collect();
            }
            b"DSOU" => {
                sound_directory = parse_directory(data, &block)
                    .with_context(|| "parsing DSOU")?
                    .into_iter()
                    .map(|(id, file_num, offset)| SoundDirEntry { sound_id: id, file_num, offset })
                    .collect();
            }
            _ => {}
        }

        pos = block.end_offset();
    }

    Ok(GameIndex { room_directory, sound_directory, room_names })
}

/// Parse RNAM block: V5 has room_no(1) + room_name(9 bytes XOR 0xFF) entries.
/// V6/V7 store no names (block is just a terminator byte).
fn parse_rnam(data: &[u8], block: &Block) -> HashMap<u8, String> {
    let mut names = HashMap::new();
    let d = &data[block.data_offset()..block.end_offset()];
    let mut pos = 0;

    while pos < d.len() {
        let room_no = d[pos];
        if room_no == 0 {
            break;
        }
        if pos + 10 > d.len() {
            break;
        }
        let name_bytes: Vec<u8> = d[pos + 1..pos + 10]
            .iter()
            .map(|b| b ^ 0xFF)
            .collect();
        let name = name_bytes
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect::<String>();
        if !name.is_empty() {
            names.insert(room_no, name);
        }
        pos += 10;
    }

    names
}

/// Parse a directory block (DROO, DSOU, DSCR, DCOS, DCHR all share the same format).
fn parse_directory(data: &[u8], block: &Block) -> Result<Vec<(usize, u8, u32)>> {
    let d = &data[block.data_offset()..block.end_offset()];
    if d.len() < 2 {
        anyhow::bail!("Directory block too small");
    }

    let num_items = u16::from_le_bytes([d[0], d[1]]) as usize;
    let expected_size = 2 + num_items + num_items * 4;
    if d.len() < expected_size {
        anyhow::bail!(
            "Directory block too small: need {} bytes, have {}",
            expected_size,
            d.len()
        );
    }

    let mut entries = Vec::new();
    for i in 0..num_items {
        let file_num = d[2 + i];
        let offset_pos = 2 + num_items + i * 4;
        let offset = u32::from_le_bytes([
            d[offset_pos],
            d[offset_pos + 1],
            d[offset_pos + 2],
            d[offset_pos + 3],
        ]);

        if file_num != 0 || offset != 0 {
            entries.push((i, file_num, offset));
        }
    }

    Ok(entries)
}

/// Parse V3/V4 index file (00.LFL).
pub fn parse_index_v3(data: &[u8]) -> Result<GameIndex> {
    let mut room_directory = Vec::new();
    let mut pos = 0;

    while pos + 6 <= data.len() {
        let size = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
        ]) as usize;
        let tag = [data[pos + 4], data[pos + 5]];

        if size < 6 || pos + size > data.len() {
            break;
        }

        let block_data = &data[pos + 6..pos + size];

        if tag == *b"0R" {
            if block_data.len() >= 2 {
                let num_items = u16::from_le_bytes([block_data[0], block_data[1]]) as usize;
                for i in 0..num_items {
                    let base = 2 + i * 5;
                    if base + 5 <= block_data.len() {
                        let file_num = block_data[base];
                        let offset = u32::from_le_bytes([
                            block_data[base + 1], block_data[base + 2],
                            block_data[base + 3], block_data[base + 4],
                        ]);
                        if file_num != 0 || offset != 0 {
                            room_directory.push(RoomDirEntry {
                                room_id: i,
                                file_num,
                                offset,
                            });
                        }
                    }
                }
            }
        }

        pos += size;
    }

    Ok(GameIndex {
        room_directory,
        sound_directory: Vec::new(),
        room_names: HashMap::new(),
    })
}
