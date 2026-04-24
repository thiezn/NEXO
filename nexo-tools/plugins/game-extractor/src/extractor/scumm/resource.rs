use super::block::{find_child, iter_children, parse_block};
use anyhow::{Context, Result};

use super::index::RoomDirEntry;
use std::path::Path;

#[derive(Debug)]
pub struct LflfEntry {
    pub room_num: u8,
    pub block: super::block::Block,
}

/// Parse the data file (.001) and return all LFLF blocks.
pub fn parse_data_file(data: &[u8]) -> Result<Vec<LflfEntry>> {
    let lecf = parse_block(data, 0).context("parsing LECF block")?;

    if &lecf.tag != b"LECF" {
        anyhow::bail!("Expected LECF block, found '{}'", lecf.tag_str());
    }

    // LOFF block contains room-to-offset mapping
    let mut room_offsets: Vec<(u8, u32)> = Vec::new();
    if let Some(loff) = find_child(data, &lecf, b"LOFF") {
        let d = &data[loff.data_offset()..loff.end_offset()];
        if !d.is_empty() {
            let num_rooms = d[0] as usize;
            for i in 0..num_rooms {
                let base = 1 + i * 5;
                if base + 5 <= d.len() {
                    let room_num = d[base];
                    let offset =
                        u32::from_le_bytes([d[base + 1], d[base + 2], d[base + 3], d[base + 4]]);
                    room_offsets.push((room_num, offset));
                }
            }
        }
    }

    let mut lflf_entries = Vec::new();
    let children = iter_children(data, &lecf);

    let mut lflf_idx = 0;
    for child in &children {
        if &child.tag == b"LFLF" {
            let room_num = room_offsets
                .iter()
                .find(|(_, off)| *off as usize == child.data_offset())
                .map(|(num, _)| *num)
                .unwrap_or_else(|| {
                    if lflf_idx < room_offsets.len() {
                        room_offsets[lflf_idx].0
                    } else {
                        lflf_idx as u8
                    }
                });

            lflf_entries.push(LflfEntry {
                room_num,
                block: child.clone(),
            });
            lflf_idx += 1;
        }
    }

    Ok(lflf_entries)
}

/// Parse V3/V4 game data from individual LFL room files.
/// Each room_id maps to its own NN.LFL file.
/// V3/V4 LFL files are NOT encrypted.
pub fn parse_v3_rooms(
    game_dir: &Path,
    room_entries: &[RoomDirEntry],
) -> Result<Vec<(u8, Vec<u8>)>> {
    let mut rooms = Vec::new();

    for entry in room_entries {
        let room_id = entry.room_id as u8;
        if room_id == 0 {
            continue; // Skip room 0 (index file)
        }

        let filename = format!("{:02}.LFL", room_id);
        let path = game_dir.join(&filename);

        let data = if path.exists() {
            std::fs::read(&path).with_context(|| format!("Failed to read {}", path.display()))?
        } else {
            let path_lower = game_dir.join(format!("{:02}.lfl", room_id));
            if !path_lower.exists() {
                continue;
            }
            std::fs::read(&path_lower)
                .with_context(|| format!("Failed to read {}", path_lower.display()))?
        };

        rooms.push((room_id, data));
    }

    Ok(rooms)
}
