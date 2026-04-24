use super::palette::Palette;
use super::version::SciVersion;
use anyhow::Result;

/// A single cel (frame) extracted from a view.
pub struct ViewCel {
    pub width: u16,
    pub height: u16,
    pub displace_x: i16,
    pub displace_y: i16,
    pub clear_key: u8,
    pub pixels: Vec<u8>,
}

/// A loop (animation direction) containing multiple cels.
pub struct ViewLoop {
    pub cels: Vec<ViewCel>,
    pub is_mirror: bool,
    pub mirror_of: Option<usize>,
}

/// Complete parsed view resource.
pub struct ViewResource {
    pub loops: Vec<ViewLoop>,
    pub embedded_palette: Option<Palette>,
}

/// Parse a view resource based on SCI version.
pub fn parse_view(data: &[u8], version: SciVersion) -> Result<ViewResource> {
    match version {
        SciVersion::Sci0
        | SciVersion::Sci01
        | SciVersion::Sci1Early
        | SciVersion::Sci1Middle
        | SciVersion::Sci1Late => parse_view_vga(data),
        SciVersion::Sci11 => parse_view_vga11(data),
        SciVersion::Sci2 | SciVersion::Sci21 => {
            parse_view_vga11(data) // SCI32 views use a similar format
        }
    }
}

/// Parse SCI0/SCI1 VGA view format.
/// Based on ScummVM GfxView::initData() kViewVga case.
fn parse_view_vga(data: &[u8]) -> Result<ViewResource> {
    if data.len() < 8 {
        anyhow::bail!("View data too small");
    }

    let loop_count = data[0] as usize;
    let flags = data[1];
    let mirror_bits = u16::from_le_bytes([data[2], data[3]]);
    // data[4..6] = version/unknown
    let palette_offset = u16::from_le_bytes([data[6], data[7]]) as usize;

    let has_palette = (flags & 0x80) != 0 || palette_offset > 0;

    // Read embedded palette if present — uses same format as palette resources
    let embedded_palette = if has_palette && palette_offset > 0 && palette_offset < data.len() {
        super::palette::parse_palette(
            &data[palette_offset..],
            super::version::SciVersion::Sci1Middle,
        )
        .ok()
        .filter(|pal| pal.iter().any(|c| c[0] != 0 || c[1] != 0 || c[2] != 0))
    } else {
        None
    };

    // Read loop offsets (2 bytes each, starting at offset 8)
    let loop_offsets_start = 8;
    let mut loops = Vec::with_capacity(loop_count);

    let mut mirror_bits_remaining = mirror_bits;

    for loop_idx in 0..loop_count {
        let lo_offset = loop_offsets_start + loop_idx * 2;
        if lo_offset + 2 > data.len() {
            break;
        }

        let is_mirror = (mirror_bits_remaining & 1) != 0;
        mirror_bits_remaining >>= 1;

        let loop_offset = u16::from_le_bytes([data[lo_offset], data[lo_offset + 1]]) as usize;

        if loop_offset + 4 > data.len() {
            loops.push(ViewLoop {
                cels: Vec::new(),
                is_mirror,
                mirror_of: None,
            });
            continue;
        }

        let cel_count = u16::from_le_bytes([data[loop_offset], data[loop_offset + 1]]) as usize;
        // data[loop_offset + 2..4] = unknown

        let mut cels = Vec::with_capacity(cel_count);

        for cel_idx in 0..cel_count {
            let cel_off_pos = loop_offset + 4 + cel_idx * 2;
            if cel_off_pos + 2 > data.len() {
                break;
            }

            // Cel offsets are ABSOLUTE from resource start (not relative to loop)
            let cel_offset =
                u16::from_le_bytes([data[cel_off_pos], data[cel_off_pos + 1]]) as usize;
            if cel_offset + 8 > data.len() {
                continue;
            }

            let width = u16::from_le_bytes([data[cel_offset], data[cel_offset + 1]]);
            let height = u16::from_le_bytes([data[cel_offset + 2], data[cel_offset + 3]]);
            let displace_x = data[cel_offset + 4] as i8 as i16;
            let displace_y = data[cel_offset + 5] as i16;
            let clear_key = data[cel_offset + 6];

            if width == 0 || height == 0 || width > 1024 || height > 1024 {
                continue;
            }

            let is_uncompressed = (flags & 0x40) != 0;
            let pixel_data_start = cel_offset + 8;

            let pixels = if is_uncompressed {
                let total = width as usize * height as usize;
                if pixel_data_start + total <= data.len() {
                    data[pixel_data_start..pixel_data_start + total].to_vec()
                } else {
                    vec![clear_key; total]
                }
            } else {
                decode_rle_vga(&data[pixel_data_start..], width, height, clear_key)?
            };

            // Mirror the displacement if this is a mirrored loop
            let final_displace_x = if is_mirror { -displace_x } else { displace_x };

            cels.push(ViewCel {
                width,
                height,
                displace_x: final_displace_x,
                displace_y,
                clear_key,
                pixels,
            });
        }

        // If mirrored, flip all cel pixels horizontally
        if is_mirror {
            for cel in &mut cels {
                let w = cel.width as usize;
                let h = cel.height as usize;
                let mut flipped = vec![0u8; cel.pixels.len()];
                for y in 0..h {
                    for x in 0..w {
                        flipped[y * w + x] = cel.pixels[y * w + (w - 1 - x)];
                    }
                }
                cel.pixels = flipped;
            }
        }

        loops.push(ViewLoop {
            cels,
            is_mirror,
            mirror_of: None,
        });
    }

    Ok(ViewResource {
        loops,
        embedded_palette,
    })
}

/// Parse SCI1.1 VGA view format.
/// Based on ScummVM GfxView::initData() kViewVga11 case.
fn parse_view_vga11(data: &[u8]) -> Result<ViewResource> {
    if data.len() < 14 {
        anyhow::bail!("SCI1.1 view data too small");
    }

    let header_size = u16::from_be_bytes([data[0], data[1]]) as usize;
    if header_size + 2 > data.len() {
        // Try LE interpretation
        let header_size_le = u16::from_le_bytes([data[0], data[1]]) as usize;
        if header_size_le + 2 <= data.len() && header_size_le > 14 {
            return parse_view_vga11_le(data);
        }
        anyhow::bail!("SCI1.1 view: invalid header size {}", header_size);
    }

    let loop_count = data[2] as usize;
    let _flags = data[3];
    // data[4..6] = version
    // data[6..8] = unknown
    let palette_offset = if data.len() >= 12 {
        u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize
    } else {
        0
    };
    let loop_size = data[12] as usize;
    let cel_size = data[13] as usize;

    let embedded_palette = if palette_offset > 0 && palette_offset < data.len() {
        super::palette::parse_view_palette(&data[palette_offset..]).ok()
    } else {
        None
    };

    let mut loops = Vec::with_capacity(loop_count);
    let loop_data_start = header_size;

    for loop_idx in 0..loop_count {
        let lo = loop_data_start + loop_idx * loop_size;
        if lo + loop_size > data.len() || loop_size < 16 {
            break;
        }

        let mirror_loop = data[lo];
        let cel_count = data[lo + 2] as usize;

        if mirror_loop != 0xFF && (mirror_loop as usize) < loop_count {
            loops.push(ViewLoop {
                cels: Vec::new(),
                is_mirror: true,
                mirror_of: Some(mirror_loop as usize),
            });
            continue;
        }

        // Cel data offset is at lo + 12 (4 bytes BE)
        let cel_data_offset = if lo + 16 <= data.len() {
            u32::from_be_bytes([data[lo + 12], data[lo + 13], data[lo + 14], data[lo + 15]])
                as usize
        } else {
            continue;
        };

        let mut cels = Vec::with_capacity(cel_count);

        for cel_idx in 0..cel_count {
            let co = cel_data_offset + cel_idx * cel_size;
            if co + cel_size > data.len() || cel_size < 32 {
                continue;
            }

            let width = u16::from_be_bytes([data[co], data[co + 1]]) as u16;
            let height = u16::from_be_bytes([data[co + 2], data[co + 3]]) as u16;
            let displace_x = i16::from_be_bytes([data[co + 4], data[co + 5]]);
            let displace_y = i16::from_be_bytes([data[co + 6], data[co + 7]]);
            let clear_key = data[co + 8];

            if width == 0 || height == 0 || width > 1024 || height > 1024 {
                continue;
            }

            // RLE offset at co + 24, literal offset at co + 28
            let rle_offset =
                u32::from_be_bytes([data[co + 24], data[co + 25], data[co + 26], data[co + 27]])
                    as usize;
            let literal_offset =
                u32::from_be_bytes([data[co + 28], data[co + 29], data[co + 30], data[co + 31]])
                    as usize;

            let pixels = if rle_offset > 0 && rle_offset < data.len() {
                let rle_data = &data[rle_offset..];
                let lit_data = if literal_offset > 0 && literal_offset < data.len() {
                    Some(&data[literal_offset..])
                } else {
                    None
                };
                decode_rle_vga11(rle_data, lit_data, width, height, clear_key)?
            } else {
                vec![clear_key; width as usize * height as usize]
            };

            cels.push(ViewCel {
                width,
                height,
                displace_x,
                displace_y,
                clear_key,
                pixels,
            });
        }

        loops.push(ViewLoop {
            cels,
            is_mirror: false,
            mirror_of: None,
        });
    }

    resolve_mirrors(&mut loops);

    Ok(ViewResource {
        loops,
        embedded_palette,
    })
}

/// Try LE interpretation for view header (some games use LE)
fn parse_view_vga11_le(data: &[u8]) -> Result<ViewResource> {
    // Fallback: try parsing as SCI0/SCI1 VGA format
    parse_view_vga(data)
}

/// Decode SCI0/SCI1 VGA RLE data.
/// Format: each byte encodes either a run or literal data.
/// Top 2 bits determine the command.
fn decode_rle_vga(rle_data: &[u8], width: u16, height: u16, clear_key: u8) -> Result<Vec<u8>> {
    let total = width as usize * height as usize;
    let mut pixels = vec![clear_key; total];
    let mut pos = 0;
    let mut out_pos = 0;

    while out_pos < total && pos < rle_data.len() {
        let cmd = rle_data[pos];
        pos += 1;

        let command = cmd & 0xC0;
        let count = (cmd & 0x3F) as usize;

        match command {
            0xC0 => {
                // Skip (transparent) — fill with clear_key
                let cnt = if count == 0 { 64 } else { count };
                out_pos += cnt;
            }
            0x80 => {
                // Fill — next byte repeated
                if pos >= rle_data.len() {
                    break;
                }
                let fill_byte = rle_data[pos];
                pos += 1;
                let cnt = if count == 0 { 64 } else { count };
                for _ in 0..cnt {
                    if out_pos >= total {
                        break;
                    }
                    pixels[out_pos] = fill_byte;
                    out_pos += 1;
                }
            }
            0x40 => {
                // Literal run (count + 64 bytes)
                let cnt = count + 64;
                for _ in 0..cnt {
                    if pos >= rle_data.len() || out_pos >= total {
                        break;
                    }
                    pixels[out_pos] = rle_data[pos];
                    pos += 1;
                    out_pos += 1;
                }
            }
            _ => {
                // 0x00: Literal run (count bytes)
                let cnt = if count == 0 { 64 } else { count };
                for _ in 0..cnt {
                    if pos >= rle_data.len() || out_pos >= total {
                        break;
                    }
                    pixels[out_pos] = rle_data[pos];
                    pos += 1;
                    out_pos += 1;
                }
            }
        }
    }

    Ok(pixels)
}

/// Decode SCI1.1 VGA RLE data with separate literal stream.
/// Public so picture.rs can reuse it for SCI1.1 bitmap pics (same RLE format).
pub fn decode_rle_vga11(
    rle_data: &[u8],
    literal_data: Option<&[u8]>,
    width: u16,
    height: u16,
    clear_key: u8,
) -> Result<Vec<u8>> {
    let total = width as usize * height as usize;
    let mut pixels = vec![clear_key; total];
    let mut rle_pos = 0;
    let mut lit_pos = 0;
    let mut out_pos = 0;

    let has_literal = literal_data.is_some();
    let lit_data = literal_data.unwrap_or(&[]);

    while out_pos < total && rle_pos < rle_data.len() {
        let cmd = rle_data[rle_pos];
        rle_pos += 1;

        let run_length = (cmd & 0x3F) as usize;

        match cmd & 0xC0 {
            0xC0 => {
                // Skip (transparent)
                out_pos += run_length;
            }
            0x80 => {
                // Fill — byte from literal stream (if available) or RLE stream
                let fill_byte = if has_literal {
                    if lit_pos >= lit_data.len() {
                        break;
                    }
                    let b = lit_data[lit_pos];
                    lit_pos += 1;
                    b
                } else {
                    if rle_pos >= rle_data.len() {
                        break;
                    }
                    let b = rle_data[rle_pos];
                    rle_pos += 1;
                    b
                };
                for _ in 0..run_length {
                    if out_pos >= total {
                        break;
                    }
                    pixels[out_pos] = fill_byte;
                    out_pos += 1;
                }
            }
            0x40 | 0x00 => {
                // Copy bytes: 0x40 = run_length + 64, 0x00 = run_length
                let cnt = if (cmd & 0xC0) == 0x40 {
                    run_length + 64
                } else {
                    run_length
                };
                if has_literal {
                    for _ in 0..cnt {
                        if lit_pos >= lit_data.len() || out_pos >= total {
                            break;
                        }
                        pixels[out_pos] = lit_data[lit_pos];
                        lit_pos += 1;
                        out_pos += 1;
                    }
                } else {
                    for _ in 0..cnt {
                        if rle_pos >= rle_data.len() || out_pos >= total {
                            break;
                        }
                        pixels[out_pos] = rle_data[rle_pos];
                        rle_pos += 1;
                        out_pos += 1;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(pixels)
}

/// Resolve mirror loops by copying from their source.
fn resolve_mirrors(loops: &mut Vec<ViewLoop>) {
    for i in 0..loops.len() {
        if let Some(mirror_src) = loops[i].mirror_of {
            if mirror_src < loops.len() && mirror_src != i && !loops[mirror_src].cels.is_empty() {
                // Clone cels from source and mirror horizontally
                let mirrored_cels: Vec<ViewCel> = loops[mirror_src]
                    .cels
                    .iter()
                    .map(|cel| {
                        let mut mirrored_pixels = vec![0u8; cel.pixels.len()];
                        for y in 0..cel.height as usize {
                            for x in 0..cel.width as usize {
                                let src_x = cel.width as usize - 1 - x;
                                mirrored_pixels[y * cel.width as usize + x] =
                                    cel.pixels[y * cel.width as usize + src_x];
                            }
                        }
                        ViewCel {
                            width: cel.width,
                            height: cel.height,
                            displace_x: -cel.displace_x,
                            displace_y: cel.displace_y,
                            clear_key: cel.clear_key,
                            pixels: mirrored_pixels,
                        }
                    })
                    .collect();
                loops[i].cels = mirrored_cels;
            }
        }
    }
}

/// Build a sprite sheet from a slice of cels (horizontal strip).
pub fn build_sprite_sheet(cels: &[&ViewCel], clear_key: u8) -> (u32, u32, Vec<u8>) {
    if cels.is_empty() {
        return (0, 0, Vec::new());
    }

    let total_width: u32 = cels.iter().map(|c| c.width as u32).sum();
    let max_height: u32 = cels.iter().map(|c| c.height as u32).max().unwrap_or(0);

    if total_width == 0 || max_height == 0 {
        return (0, 0, Vec::new());
    }

    let mut sheet = vec![clear_key; (total_width * max_height) as usize];
    let mut x_offset = 0u32;

    for cel in cels {
        let cw = cel.width as u32;
        let ch = cel.height as u32;
        for y in 0..ch {
            for x in 0..cw {
                let src_idx = (y * cw + x) as usize;
                let dst_idx = (y * total_width + x_offset + x) as usize;
                if src_idx < cel.pixels.len() && dst_idx < sheet.len() {
                    sheet[dst_idx] = cel.pixels[src_idx];
                }
            }
        }
        x_offset += cw;
    }

    (total_width, max_height, sheet)
}
