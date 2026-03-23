use super::block::{self, Block};
use super::image_decode;
use super::version::ScummVersion;
use anyhow::{Result, bail};

/// Decoded costume frame (single cel)
pub struct CostumeFrame {
    pub width: u16,
    pub height: u16,
    pub rel_x: i16,
    pub rel_y: i16,
    pub move_x: i16,
    pub move_y: i16,
    pub pixels: Vec<u8>, // palette indices, width*height
}

/// A costume animation: sequence of frame indices per limb
pub struct CostumeAnimation {
    pub anim_index: usize,
    pub name: String,
    /// Per-limb frame sequences. Each entry is (limb_index, vec of frame indices).
    pub limb_frames: Vec<(usize, Vec<usize>)>,
}

/// Full costume info extracted from a COST or AKOS block
pub struct CostumeInfo {
    pub costume_id: usize,
    pub palette: Vec<u8>,
    pub frames: Vec<CostumeFrame>,
    pub animations: Vec<CostumeAnimation>,
    pub mirror: bool,
}

// Animation name mapping: groups of 4 (west, east, south, north)
const ANIM_NAMES: &[&str] = &["init", "walk", "stand", "talk_start", "talk_stop"];

fn anim_group_name(frame: usize, dir: usize) -> String {
    let dir_name = match dir {
        0 => "west",
        1 => "east",
        2 => "south",
        3 => "north",
        _ => "unknown",
    };
    let group = frame;
    let group_name = if group < ANIM_NAMES.len() {
        ANIM_NAMES[group].to_string()
    } else {
        format!("anim_{:02}", group)
    };
    format!("{}_{}", group_name, dir_name)
}

// ─── COST format (V5/V6) ───────────────────────────────────────────────

/// Extract costume data from a COST block.
/// `cost_data` is the raw data of the COST block (after the 8-byte block tag+size header).
pub fn parse_cost(cost_data: &[u8], version: ScummVersion) -> Result<CostumeInfo> {
    // V5: baseptr overlaps with block header bytes before resource data.
    // In our case, cost_data starts at resource data (after 8-byte block header).
    // V5: num_anim = cost_data[0], format = cost_data[1], palette = cost_data[2..]
    // V6: cost_data[0..3] = LE size, [4..5] = "CO", [6] = num_anim, [7] = format, [8..] = palette
    //
    // For offset calculations: V5 baseptr = block_start + 2, which is resource_data - 6.
    // V6 baseptr = block_start + 8 = resource_data.
    // All internal offsets are relative to baseptr.

    let (num_anim, format_byte, palette_start, baseptr_offset): (usize, u8, usize, usize) =
        match version {
            ScummVersion::V5 => {
                let na = cost_data[0] as usize;
                let fb = cost_data[1];
                (na, fb, 2, 6) // baseptr is 6 bytes before resource_data
            }
            ScummVersion::V6 | ScummVersion::V7 => {
                if cost_data.len() < 8 {
                    bail!("COST block too small");
                }
                let na = cost_data[6] as usize;
                let fb = cost_data[7];
                (na, fb, 8, 0) // baseptr == resource_data
            }
            _ => bail!("COST not supported for {:?}", version),
        };

    let mirror = (format_byte & 0x80) != 0;
    let palette_mode = format_byte & 0x7F;

    let (num_colors, shift, mask): (usize, u8, u8) = match palette_mode {
        0x58 => (16, 4, 0x0F),
        0x59 => (32, 3, 0x07),
        0x60 => (16, 4, 0x0F),
        0x61 => (32, 3, 0x07),
        _ => bail!("Unknown COST palette mode 0x{:02x}", palette_mode),
    };

    let palette: Vec<u8> = cost_data[palette_start..palette_start + num_colors].to_vec();

    // anim_cmds_offset (relative to baseptr)
    let aco_pos = palette_start + num_colors;
    if aco_pos + 2 > cost_data.len() {
        bail!("COST too small for anim_cmds_offset");
    }
    let anim_cmds_offset =
        u16::from_le_bytes([cost_data[aco_pos], cost_data[aco_pos + 1]]) as usize;

    // frame_offsets: 16 limb offsets (relative to baseptr)
    let fo_pos = aco_pos + 2;
    if fo_pos + 32 > cost_data.len() {
        bail!("COST too small for frame_offsets");
    }
    let mut limb_frame_offsets = [0u16; 16];
    for i in 0..16 {
        limb_frame_offsets[i] =
            u16::from_le_bytes([cost_data[fo_pos + i * 2], cost_data[fo_pos + i * 2 + 1]]);
    }

    // anim data offsets (relative to baseptr)
    let do_pos = fo_pos + 32;

    // Helper: convert baseptr-relative offset to cost_data index
    let bp_to_idx = |bp_off: usize| -> usize {
        if bp_off >= baseptr_offset {
            bp_off - baseptr_offset
        } else {
            0 // shouldn't happen
        }
    };

    let anim_cmds_idx = bp_to_idx(anim_cmds_offset);

    // Collect all unique frames we need to decode
    // frame_key: (limb, picture_code) -> frame_index
    let mut frames: Vec<CostumeFrame> = Vec::new();
    let mut frame_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();

    // For each limb, build the picture table
    // frameptr = baseptr + limb_frame_offsets[limb]
    // picture_data_offset = READ_LE_UINT16(frameptr + code * 2) (baseptr-relative)

    let cost_data_len = cost_data.len();

    let decode_frame = |code: usize, limb: usize| -> Result<CostumeFrame> {
        let fp_idx = bp_to_idx(limb_frame_offsets[limb] as usize);
        let pic_off_pos = fp_idx + code * 2;
        if pic_off_pos + 2 > cost_data_len {
            bail!("Frame offset out of bounds");
        }
        let pic_bp_off =
            u16::from_le_bytes([cost_data[pic_off_pos], cost_data[pic_off_pos + 1]]) as usize;
        // Validate the picture offset is within reasonable range
        if pic_bp_off < baseptr_offset || pic_bp_off > cost_data_len + baseptr_offset {
            bail!("Invalid picture offset {}", pic_bp_off);
        }
        let pic_idx = pic_bp_off - baseptr_offset;
        if pic_idx + 12 > cost_data_len {
            bail!("Frame data out of bounds");
        }

        let width = u16::from_le_bytes([cost_data[pic_idx], cost_data[pic_idx + 1]]);
        let height = u16::from_le_bytes([cost_data[pic_idx + 2], cost_data[pic_idx + 3]]);
        // Sanity check dimensions
        if width > 1024 || height > 1024 || (width as usize * height as usize) > 500_000 {
            bail!("Unreasonable frame dimensions {}x{}", width, height);
        }
        let rel_x = i16::from_le_bytes([cost_data[pic_idx + 4], cost_data[pic_idx + 5]]);
        let rel_y = i16::from_le_bytes([cost_data[pic_idx + 6], cost_data[pic_idx + 7]]);
        let move_x = i16::from_le_bytes([cost_data[pic_idx + 8], cost_data[pic_idx + 9]]);
        let move_y = i16::from_le_bytes([cost_data[pic_idx + 10], cost_data[pic_idx + 11]]);

        // For format 0x60/0x61 (V6), there are 2 extra indirection bytes after CostumeInfo
        let rle_start = if palette_mode == 0x60 || palette_mode == 0x61 {
            let extra_start = pic_idx + 12;
            if extra_start + 2 > cost_data_len {
                bail!("RLE data out of bounds");
            }
            let ex1 = cost_data[extra_start] as usize;
            let ex2 = cost_data[extra_start + 1] as usize;
            if ex1 != 0xFF || ex2 != 0xFF {
                // Follow indirection: ex1 indexes into frame_offsets,
                // then ex2 indexes into that limb's picture table
                if ex1 < 16 {
                    let fo = limb_frame_offsets[ex1] as usize;
                    let fo_idx = bp_to_idx(fo);
                    let inner_pos = fo_idx + ex2 * 2;
                    if inner_pos + 2 <= cost_data_len {
                        let pic_off =
                            u16::from_le_bytes([cost_data[inner_pos], cost_data[inner_pos + 1]])
                                as usize;
                        let resolved = bp_to_idx(pic_off) + 14;
                        if resolved < cost_data_len {
                            resolved
                        } else {
                            extra_start + 2
                        }
                    } else {
                        extra_start + 2
                    }
                } else {
                    extra_start + 2
                }
            } else {
                extra_start + 2
            }
        } else {
            pic_idx + 12
        };
        if rle_start >= cost_data_len {
            bail!("RLE data out of bounds");
        }
        let pixels = decode_cost_rle(
            &cost_data[rle_start..],
            width as usize,
            height as usize,
            shift,
            mask,
        )?;

        Ok(CostumeFrame {
            width,
            height,
            rel_x,
            rel_y,
            move_x,
            move_y,
            pixels,
        })
    };

    // Parse animations
    let mut animations: Vec<CostumeAnimation> = Vec::new();

    for anim_idx in 0..=num_anim {
        // Each anim corresponds to a direction variant: anim_idx = frame * 4 + dir
        let frame_group = anim_idx / 4;
        let dir = anim_idx % 4;

        if do_pos + anim_idx * 2 + 2 > cost_data.len() {
            break;
        }
        let anim_offset = u16::from_le_bytes([
            cost_data[do_pos + anim_idx * 2],
            cost_data[do_pos + anim_idx * 2 + 1],
        ]) as usize;

        if anim_offset == 0 {
            continue;
        }

        let anim_data_idx = bp_to_idx(anim_offset);
        if anim_data_idx + 2 > cost_data.len() {
            continue;
        }

        let limb_mask =
            u16::from_le_bytes([cost_data[anim_data_idx], cost_data[anim_data_idx + 1]]);
        let mut r = anim_data_idx + 2;
        let mut limb_frames_list: Vec<(usize, Vec<usize>)> = Vec::new();

        for bit in 0..16 {
            if limb_mask & (0x8000 >> bit) == 0 {
                continue;
            }
            if r + 2 > cost_data.len() {
                break;
            }
            let j = u16::from_le_bytes([cost_data[r], cost_data[r + 1]]) as usize;
            r += 2;
            if j == 0xFFFF {
                continue; // limb disabled
            }
            if r >= cost_data.len() {
                break;
            }
            let extra = cost_data[r];
            r += 1;
            let len = (extra & 0x7F) as usize;

            // Collect frame indices for this limb
            let mut limb_frame_indices = Vec::new();
            for cmd_pos in j..=j + len {
                if anim_cmds_idx + cmd_pos >= cost_data.len() {
                    break;
                }
                let cmd = cost_data[anim_cmds_idx + cmd_pos];
                let code = (cmd & 0x7F) as usize;
                if code >= 0x79 {
                    continue; // special commands (stop/start/hide)
                }

                let key = (bit, code);
                let frame_idx = if let Some(&idx) = frame_map.get(&key) {
                    idx
                } else {
                    match decode_frame(code, bit) {
                        Ok(frame) => {
                            let idx = frames.len();
                            frames.push(frame);
                            frame_map.insert(key, idx);
                            idx
                        }
                        Err(_) => continue,
                    }
                };
                limb_frame_indices.push(frame_idx);
            }

            if !limb_frame_indices.is_empty() {
                limb_frames_list.push((bit, limb_frame_indices));
            }
        }

        if !limb_frames_list.is_empty() {
            animations.push(CostumeAnimation {
                anim_index: anim_idx,
                name: anim_group_name(frame_group, dir),
                limb_frames: limb_frames_list,
            });
        }
    }

    Ok(CostumeInfo {
        costume_id: 0,
        palette,
        frames,
        animations,
        mirror,
    })
}

/// Column-wise RLE decoder for COST frames
fn decode_cost_rle(
    data: &[u8],
    width: usize,
    height: usize,
    shift: u8,
    mask: u8,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let mut pixels = vec![0u8; width * height];
    let mut x = 0usize;
    let mut y = 0usize;
    let mut pos = 0usize;

    while x < width && pos < data.len() {
        let byte = data[pos];
        pos += 1;
        let color = byte >> shift;
        let mut rep = (byte & mask) as usize;
        if rep == 0 {
            if pos >= data.len() {
                break;
            }
            rep = data[pos] as usize;
            pos += 1;
        }

        for _ in 0..rep {
            if x < width && y < height {
                pixels[y * width + x] = color;
            }
            y += 1;
            if y >= height {
                y = 0;
                x += 1;
                if x >= width {
                    break;
                }
            }
        }
    }

    Ok(pixels)
}

// ─── AKOS format (V7) ──────────────────────────────────────────────────

/// Extract costume data from an AKOS block.
/// `data` is the full game data, `akos_block` is the AKOS block.
pub fn parse_akos(data: &[u8], akos_block: &Block) -> Result<CostumeInfo> {
    // Find sub-blocks
    let akhd = block::find_child(data, akos_block, b"AKHD")
        .ok_or_else(|| anyhow::anyhow!("AKOS missing AKHD"))?;
    let akof = block::find_child(data, akos_block, b"AKOF");
    let akci = block::find_child(data, akos_block, b"AKCI");
    let akcd = block::find_child(data, akos_block, b"AKCD");
    let akpl = block::find_child(data, akos_block, b"AKPL");
    let akch = block::find_child(data, akos_block, b"AKCH");

    // Parse AKHD header
    let hd = &data[akhd.data_offset()..akhd.end_offset()];
    if hd.len() < 12 {
        bail!("AKHD too small");
    }
    let _version = u16::from_le_bytes([hd[0], hd[1]]);
    let costume_flags = u16::from_le_bytes([hd[2], hd[3]]);
    let chore_count = u16::from_le_bytes([hd[4], hd[5]]) as usize;
    let cels_count = u16::from_le_bytes([hd[6], hd[7]]) as usize;
    let codec = u16::from_le_bytes([hd[8], hd[9]]);
    let _layer_count = u16::from_le_bytes([hd[10], hd[11]]);

    let mirror = (costume_flags & 1) != 0;
    let many_dirs = (costume_flags & 2) != 0;
    let dirs = if many_dirs { 8 } else { 4 };

    // Parse palette
    let palette = if let Some(ref pl) = akpl {
        data[pl.data_offset()..pl.end_offset()].to_vec()
    } else {
        Vec::new()
    };

    // Parse frames from AKOF + AKCI + AKCD
    let mut frames: Vec<CostumeFrame> = Vec::new();

    if let (Some(ref of_block), Some(ref ci_block), Some(ref cd_block)) = (akof, akci, akcd) {
        let of_data = &data[of_block.data_offset()..of_block.end_offset()];
        let ci_data = &data[ci_block.data_offset()..ci_block.end_offset()];
        let cd_data = &data[cd_block.data_offset()..cd_block.end_offset()];

        for i in 0..cels_count {
            let of_pos = i * 6;
            if of_pos + 6 > of_data.len() {
                break;
            }
            let cd_offset = u32::from_le_bytes([
                of_data[of_pos],
                of_data[of_pos + 1],
                of_data[of_pos + 2],
                of_data[of_pos + 3],
            ]) as usize;
            let ci_offset = u16::from_le_bytes([of_data[of_pos + 4], of_data[of_pos + 5]]) as usize;

            if ci_offset + 12 > ci_data.len() {
                continue;
            }

            let width = u16::from_le_bytes([ci_data[ci_offset], ci_data[ci_offset + 1]]);
            let height = u16::from_le_bytes([ci_data[ci_offset + 2], ci_data[ci_offset + 3]]);
            let rel_x = i16::from_le_bytes([ci_data[ci_offset + 4], ci_data[ci_offset + 5]]);
            let rel_y = i16::from_le_bytes([ci_data[ci_offset + 6], ci_data[ci_offset + 7]]);
            let move_x = i16::from_le_bytes([ci_data[ci_offset + 8], ci_data[ci_offset + 9]]);
            let move_y = i16::from_le_bytes([ci_data[ci_offset + 10], ci_data[ci_offset + 11]]);

            if width == 0 || height == 0 {
                frames.push(CostumeFrame {
                    width,
                    height,
                    rel_x,
                    rel_y,
                    move_x,
                    move_y,
                    pixels: Vec::new(),
                });
                continue;
            }

            let frame_data = if cd_offset < cd_data.len() {
                &cd_data[cd_offset..]
            } else {
                frames.push(CostumeFrame {
                    width,
                    height,
                    rel_x,
                    rel_y,
                    move_x,
                    move_y,
                    pixels: vec![0; (width as usize) * (height as usize)],
                });
                continue;
            };

            let pixels = match codec {
                1 => {
                    // BYLE RLE - column-wise, same as COST
                    // Determine shift/mask from palette size
                    let (shift, mask) = if palette.len() <= 16 {
                        (4u8, 0x0Fu8)
                    } else if palette.len() <= 32 {
                        (3, 0x07)
                    } else {
                        (2, 0x03) // 64 colors
                    };
                    decode_cost_rle(frame_data, width as usize, height as usize, shift, mask)
                        .unwrap_or_else(|_| vec![0; (width as usize) * (height as usize)])
                }
                5 => {
                    // BOMP/CDAT RLE
                    image_decode::decode_bomp(frame_data, width as usize, height as usize)
                        .unwrap_or_else(|_| vec![0; (width as usize) * (height as usize)])
                }
                16 => {
                    // MAJ-MIN bit-delta codec
                    decode_akos_majmin(frame_data, width as usize, height as usize)
                        .unwrap_or_else(|_| vec![0; (width as usize) * (height as usize)])
                }
                _ => {
                    vec![0; (width as usize) * (height as usize)]
                }
            };

            frames.push(CostumeFrame {
                width,
                height,
                rel_x,
                rel_y,
                move_x,
                move_y,
                pixels,
            });
        }
    }

    // Parse animations from AKCH
    let mut animations: Vec<CostumeAnimation> = Vec::new();

    if let Some(ref ch_block) = akch {
        let ch_data = &data[ch_block.data_offset()..ch_block.end_offset()];

        for anim_idx in 0..chore_count {
            if anim_idx * 2 + 2 > ch_data.len() {
                break;
            }
            let offs =
                u16::from_le_bytes([ch_data[anim_idx * 2], ch_data[anim_idx * 2 + 1]]) as usize;
            if offs == 0 || offs + 2 > ch_data.len() {
                continue;
            }

            let frame_group = anim_idx / dirs;
            let dir = anim_idx % dirs;

            let limb_mask = u16::from_le_bytes([ch_data[offs], ch_data[offs + 1]]);
            let mut r = offs + 2;
            let mut limb_frames_list: Vec<(usize, Vec<usize>)> = Vec::new();

            for bit in 0..16 {
                if limb_mask & (0x8000 >> bit) == 0 {
                    continue;
                }
                if r >= ch_data.len() {
                    break;
                }
                let code = ch_data[r];
                r += 1;

                match code {
                    1 | 4 | 5 => {
                        // Empty, stopped, or started - no frame data
                    }
                    _ => {
                        if r + 4 > ch_data.len() {
                            break;
                        }
                        let start = u16::from_le_bytes([ch_data[r], ch_data[r + 1]]) as usize;
                        r += 2;
                        let len = u16::from_le_bytes([ch_data[r], ch_data[r + 1]]) as usize;
                        r += 2;

                        // The AKSQ sequence at positions start..start+len contains frame indices
                        // For simplicity, we just collect the cel indices referenced
                        let frame_indices =
                            collect_aksq_frames(data, akos_block, start, len, cels_count);
                        if !frame_indices.is_empty() {
                            limb_frames_list.push((bit, frame_indices));
                        }
                    }
                }
            }

            if !limb_frames_list.is_empty() {
                let name = if many_dirs {
                    let dir_name = match dir {
                        0 => "west",
                        1 => "northwest",
                        2 => "north",
                        3 => "northeast",
                        4 => "east",
                        5 => "southeast",
                        6 => "south",
                        7 => "southwest",
                        _ => "unknown",
                    };
                    let group_name = if frame_group < ANIM_NAMES.len() {
                        ANIM_NAMES[frame_group].to_string()
                    } else {
                        format!("anim_{:02}", frame_group)
                    };
                    format!("{}_{}", group_name, dir_name)
                } else {
                    anim_group_name(frame_group, dir)
                };

                animations.push(CostumeAnimation {
                    anim_index: anim_idx,
                    name,
                    limb_frames: limb_frames_list,
                });
            }
        }
    }

    Ok(CostumeInfo {
        costume_id: 0,
        palette,
        frames,
        animations,
        mirror,
    })
}

/// Collect frame (cel) indices from AKSQ data for a given animation range.
fn collect_aksq_frames(
    data: &[u8],
    akos_block: &Block,
    start: usize,
    len: usize,
    max_cels: usize,
) -> Vec<usize> {
    let aksq = match block::find_child(data, akos_block, b"AKSQ") {
        Some(b) => b,
        None => return Vec::new(),
    };
    let sq_data = &data[aksq.data_offset()..aksq.end_offset()];
    let mut indices = Vec::new();
    let mut pos = start;
    let end = start + len;

    while pos < end && pos < sq_data.len() {
        let byte = sq_data[pos];
        if byte & 0x80 != 0 {
            // Multi-byte code
            if pos + 1 >= sq_data.len() {
                break;
            }
            let code = ((byte as u16) << 8) | sq_data[pos + 1] as u16;
            pos += 2;
            if code & 0xC000 == 0xC000 {
                // Command - skip its arguments
                pos = skip_aksq_command(sq_data, code, pos);
            } else {
                // Frame index
                let cel = (code & 0x0FFF) as usize;
                if cel < max_cels {
                    indices.push(cel);
                }
            }
        } else {
            // Single byte frame index
            let cel = byte as usize;
            if cel < max_cels {
                indices.push(cel);
            }
            pos += 1;
        }
    }

    indices
}

/// Skip past an AKSQ command's arguments
fn skip_aksq_command(_data: &[u8], code: u16, mut pos: usize) -> usize {
    match code {
        0xC001 | 0xC050 | 0xC060 | 0xC061 | 0xC09F | 0xC0FF => {
            // No arguments
        }
        0xC010 => {
            pos += 3;
        } // SetVar: word + byte
        0xC015 | 0xC042 => {
            pos += 1;
        } // StartSound: byte
        0xC016 | 0xC017 => {
            pos += 3;
        } // IfVarSound: word + byte
        0xC018 | 0xC019 => {
            pos += 3;
        } // IfSound: word + byte
        0xC030 => {
            pos += 2;
        } // Jump: word
        0xC031 => {
            pos += 3;
        } // JumpIfSet: word + byte
        0xC040 => {
            pos += 3;
        } // AddVar: word + byte
        0xC044 => {
            pos += 1;
        } // StartVarSound: byte
        0xC080 | 0xC081 => {
            pos += 1;
        } // StartAnim/StartVarAnim: byte
        0xC082 => {
            pos += 5;
        } // Random: word + word + byte
        0xC083 => {
            pos += 1;
        } // SetActorClip: byte
        0xC084 => {
            pos += 2;
        } // StartAnimInActor: byte + byte
        0xC085 => {
            pos += 4;
        } // SetVarInActor: byte + byte + word
        0xC086 => {} // HideActor
        0xC087 => {
            pos += 4;
        } // SetDrawOffs: word + word
        0xC088 => {
            pos += 1;
        } // JumpTable: byte
        0xC08A => {
            pos += 2;
        } // Flip: word
        0xC0A1 | 0xC0A2 => {
            pos += 2;
        } // JumpIfTalking: word
        _ => {
            // Unknown command, try to skip conservatively
            if code >= 0xC070 && code <= 0xC075 {
                pos += 5; // Conditional: word + word + byte
            } else if code >= 0xC090 && code <= 0xC095 {
                pos += 5; // Skip conditional
            } else {
                // Skip 2 bytes as conservative estimate
                pos += 2;
            }
        }
    }
    pos
}

/// Decode AKOS codec 16 (MAJ-MIN / bit-delta)
fn decode_akos_majmin(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    if data.len() < 2 || width == 0 || height == 0 {
        return Ok(vec![0; width * height]);
    }

    let total = width * height;
    let mut pixels = vec![0u8; total];
    let bpp = data[0];
    let mut color = data[1];
    let mut repeat: i32 = 0;

    // Bit reader
    let mut bit_pos: usize = 16; // start after 2 header bytes (in bits)

    let read_bit = |bit_pos: &mut usize| -> u8 {
        let byte_idx = *bit_pos / 8;
        let bit_idx = *bit_pos % 8;
        *bit_pos += 1;
        if byte_idx < data.len() {
            (data[byte_idx] >> bit_idx) & 1
        } else {
            0
        }
    };

    let read_bits = |bit_pos: &mut usize, n: u8| -> u8 {
        let mut val = 0u8;
        for i in 0..n {
            val |= read_bit(bit_pos) << i;
        }
        val
    };

    for pos in 0..total {
        pixels[pos] = color;

        if repeat > 0 {
            repeat -= 1;
        } else {
            let control = read_bit(&mut bit_pos);
            if control != 0 {
                let control2 = read_bit(&mut bit_pos);
                if control2 != 0 {
                    let delta = read_bits(&mut bit_pos, 3);
                    if delta != 4 {
                        color = color.wrapping_add(delta).wrapping_sub(4);
                    } else {
                        repeat = read_bits(&mut bit_pos, 8) as i32 - 1;
                    }
                } else {
                    color = read_bits(&mut bit_pos, bpp);
                }
            }
        }
    }

    Ok(pixels)
}

// ─── Sprite sheet generation ─────────────────────────────────────────────

/// Build a horizontal sprite sheet from a list of frames.
/// Returns (width, height, pixels) where pixels are palette indices.
pub fn build_sprite_sheet(frames: &[&CostumeFrame]) -> (u32, u32, Vec<u8>) {
    if frames.is_empty() {
        return (0, 0, Vec::new());
    }

    let max_height = frames.iter().map(|f| f.height as u32).max().unwrap_or(0);
    let total_width: u32 = frames.iter().map(|f| f.width as u32).sum();

    if total_width == 0 || max_height == 0 {
        return (0, 0, Vec::new());
    }

    let mut sheet = vec![0u8; (total_width * max_height) as usize];
    let mut x_offset: u32 = 0;

    for frame in frames {
        let fw = frame.width as u32;
        let fh = frame.height as u32;
        for y in 0..fh {
            for x in 0..fw {
                let src_idx = (y * fw + x) as usize;
                let dst_idx = (y * total_width + x_offset + x) as usize;
                if src_idx < frame.pixels.len() && dst_idx < sheet.len() {
                    sheet[dst_idx] = frame.pixels[src_idx];
                }
            }
        }
        x_offset += fw;
    }

    (total_width, max_height, sheet)
}

fn find_blocks_by_tag(data: &[u8], parent: &Block, tag: &[u8; 4]) -> Vec<(usize, Block)> {
    block::iter_children(data, parent)
        .into_iter()
        .filter(|child| &child.tag == tag)
        .enumerate()
        .map(|(i, b)| (i, b))
        .collect()
}

/// Find all COST blocks inside a ROOM block (V5/V6)
pub fn find_cost_blocks(data: &[u8], room_block: &Block) -> Vec<(usize, Block)> {
    find_blocks_by_tag(data, room_block, b"COST")
}

/// Find all AKOS blocks inside a ROOM block (V7)
pub fn find_akos_blocks(data: &[u8], room_block: &Block) -> Vec<(usize, Block)> {
    find_blocks_by_tag(data, room_block, b"AKOS")
}
