use anyhow::{Result, bail};
use super::block::{Block, find_child, find_all_children, find_child_with_prefix, find_all_with_prefix};
use super::image_decode;
use super::version::ScummVersion;
use crate::extractor::common::output::PaletteImage;

#[derive(Clone, Copy)]
pub struct Palette {
    pub colors: [(u8, u8, u8); 256],
}

pub struct RoomHeader {
    pub width: u16,
    pub height: u16,
}

pub struct DecodedImage {
    pub width: u16,
    pub height: u16,
    pub pixels: Vec<u8>,
    pub palette: Palette,
}

impl PaletteImage for DecodedImage {
    fn width(&self) -> u16 { self.width }
    fn height(&self) -> u16 { self.height }
    fn pixels(&self) -> &[u8] { &self.pixels }
    fn palette_color(&self, index: u8) -> (u8, u8, u8) {
        self.palette.colors[index as usize]
    }
}

pub struct ObjectImage {
    pub obj_id: u16,
    pub state: u8,
    pub image: DecodedImage,
}

pub struct ObjectMeta {
    pub obj_id: u16,
    pub name: Option<String>,
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    pub num_states: u8,
    pub parent: u8,
}

/// Extract the room background image from a ROOM block.
pub fn extract_background(data: &[u8], room_block: &Block, version: ScummVersion) -> Result<Option<DecodedImage>> {
    let header = parse_room_header(data, room_block, version)?;
    let palette = extract_palette(data, room_block)?;

    let rmim = match find_child(data, room_block, b"RMIM") {
        Some(b) => b,
        None => return Ok(None),
    };

    let im_block = match find_child_with_prefix(data, &rmim, b"IM") {
        Some(b) => b,
        None => return Ok(None),
    };

    let pixels = decode_image_block(data, &im_block, header.width as usize, header.height as usize)?;

    Ok(Some(DecodedImage {
        width: header.width,
        height: header.height,
        pixels,
        palette,
    }))
}

/// Extract all object images from a ROOM block.
pub fn extract_objects(data: &[u8], room_block: &Block, version: ScummVersion) -> Result<Vec<ObjectImage>> {
    let obim_blocks = find_all_children(data, room_block, b"OBIM");
    if obim_blocks.is_empty() {
        return Ok(Vec::new());
    }

    let palette = extract_palette(data, room_block)?;
    let mut results = Vec::new();

    for obim in &obim_blocks {
        match extract_object_images(data, obim, &palette, version) {
            Ok(images) => results.extend(images),
            Err(_) => {}
        }
    }

    Ok(results)
}

/// Extract object metadata from OBIM and OBCD blocks.
pub fn extract_object_metadata(data: &[u8], room_block: &Block, version: ScummVersion) -> Vec<ObjectMeta> {
    let obim_blocks = find_all_children(data, room_block, b"OBIM");
    let obcd_blocks = find_all_children(data, room_block, b"OBCD");

    let mut results = Vec::new();

    for obim in &obim_blocks {
        if let Some(imhd) = find_child(data, obim, b"IMHD") {
            let d = &data[imhd.data_offset()..imhd.end_offset()];
            if d.len() < 16 {
                continue;
            }

            // V7 IMHD: version(4) + obj_id(2) + image_count(2) + x_pos(2) + y_pos(2) + width(2) + height(2)
            // V5/V6 IMHD: obj_id(2) + image_count(2) + unk(2) + flags(1) + unk1(1) + unk2(4) + width(2) + height(2)
            let (obj_id, im_x, im_y) = match version {
                ScummVersion::V7 => {
                    if d.len() < 16 { continue; }
                    (
                        u16::from_le_bytes([d[4], d[5]]),
                        u16::from_le_bytes([d[8], d[9]]),
                        u16::from_le_bytes([d[10], d[11]]),
                    )
                }
                _ => (u16::from_le_bytes([d[0], d[1]]), 0u16, 0u16),
            };
            let im_width = u16::from_le_bytes([d[12], d[13]]);
            let im_height = u16::from_le_bytes([d[14], d[15]]);

            // Count states from IM** blocks
            let im_blocks = find_all_with_prefix(data, obim, b"IM");
            let num_states = im_blocks.iter().filter(|b| &b.tag != b"IMHD").count() as u8;

            // Look for matching OBCD block for this object
            let mut name = None;
            let mut x: u16 = im_x;
            let mut y: u16 = im_y;
            let mut cdhd_w: u16 = im_width;
            let mut cdhd_h: u16 = im_height;
            let mut parent: u8 = 0;

            for obcd in &obcd_blocks {
                if let Some(cdhd) = find_child(data, obcd, b"CDHD") {
                    let cd = &data[cdhd.data_offset()..cdhd.end_offset()];

                    // V7 CDHD: version(4) + obj_id(2) + parent(1) + parentstate(1)
                    let cd_obj_id = match version {
                        ScummVersion::V7 => {
                            if cd.len() < 8 { continue; }
                            u16::from_le_bytes([cd[4], cd[5]])
                        }
                        _ => {
                            if cd.len() < 2 { continue; }
                            u16::from_le_bytes([cd[0], cd[1]])
                        }
                    };
                    if cd_obj_id != obj_id {
                        continue;
                    }

                    match version {
                        ScummVersion::V5 => {
                            if cd.len() >= 9 {
                                x = cd[2] as u16 * 8;
                                y = cd[3] as u16 * 8;
                                cdhd_w = cd[4] as u16 * 8;
                                cdhd_h = cd[5] as u16;
                                parent = cd[7];
                            }
                        }
                        ScummVersion::V6 => {
                            if cd.len() >= 13 {
                                x = u16::from_le_bytes([cd[2], cd[3]]);
                                y = u16::from_le_bytes([cd[4], cd[5]]);
                                cdhd_w = u16::from_le_bytes([cd[6], cd[7]]);
                                cdhd_h = u16::from_le_bytes([cd[8], cd[9]]);
                                parent = cd[11];
                            }
                        }
                        ScummVersion::V7 => {
                            // V7 CDHD has no x/y/w/h — use IMHD values
                            if cd.len() >= 8 {
                                parent = cd[6];
                            }
                        }
                        _ => {}
                    }

                    // Look for OBNA (object name) inside OBCD
                    if let Some(obna) = find_child(data, obcd, b"OBNA") {
                        let nd = &data[obna.data_offset()..obna.end_offset()];
                        if let Some(end) = nd.iter().position(|&b| b == 0) {
                            name = String::from_utf8(nd[..end].to_vec()).ok();
                        } else if !nd.is_empty() {
                            name = String::from_utf8_lossy(nd).to_string().into();
                        }
                    }
                    break;
                }
            }

            results.push(ObjectMeta {
                obj_id,
                name,
                x,
                y,
                width: cdhd_w,
                height: cdhd_h,
                num_states,
                parent,
            });
        }
    }

    results
}

/// Get room header info
pub fn get_room_header(data: &[u8], room_block: &Block, version: ScummVersion) -> Result<RoomHeader> {
    parse_room_header(data, room_block, version)
}

/// Get palette type string for metadata
pub fn get_palette_type(data: &[u8], room_block: &Block) -> &'static str {
    if find_child(data, room_block, b"CLUT").is_some() {
        "CLUT"
    } else if find_child(data, room_block, b"PALS").is_some() {
        "APAL"
    } else {
        "grayscale"
    }
}

fn extract_object_images(data: &[u8], obim: &Block, palette: &Palette, version: ScummVersion) -> Result<Vec<ObjectImage>> {
    let imhd = find_child(data, obim, b"IMHD")
        .ok_or_else(|| anyhow::anyhow!("OBIM missing IMHD"))?;

    let d = &data[imhd.data_offset()..imhd.end_offset()];
    if d.len() < 16 {
        bail!("IMHD too small");
    }

    let obj_id = match version {
        ScummVersion::V7 => u16::from_le_bytes([d[4], d[5]]),
        _ => u16::from_le_bytes([d[0], d[1]]),
    };
    let width = u16::from_le_bytes([d[12], d[13]]);
    let height = u16::from_le_bytes([d[14], d[15]]);

    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    let im_blocks = find_all_with_prefix(data, obim, b"IM");

    let mut state: u8 = 0;
    for im_block in &im_blocks {
        if &im_block.tag == b"IMHD" {
            continue;
        }

        match decode_image_block(data, im_block, width as usize, height as usize) {
            Ok(pixels) => {
                results.push(ObjectImage {
                    obj_id,
                    state,
                    image: DecodedImage {
                        width,
                        height,
                        pixels,
                        palette: *palette,
                    },
                });
            }
            Err(_) => {}
        }
        state += 1;
    }

    Ok(results)
}

fn decode_image_block(data: &[u8], im_block: &Block, width: usize, height: usize) -> Result<Vec<u8>> {
    // Try SMAP first, fall back to BOMP for V6/V7 objects
    let smap = match find_child(data, im_block, b"SMAP") {
        Some(s) => s,
        None => {
            // Try BOMP fallback
            if let Some(bomp) = find_child(data, im_block, b"BOMP") {
                let bomp_data = &data[bomp.data_offset()..bomp.end_offset()];
                return image_decode::decode_bomp(bomp_data, width, height);
            }
            bail!("Image block '{}' missing SMAP and BOMP", im_block.tag_str());
        }
    };

    let smap_block = &data[smap.offset..smap.end_offset()];
    let strip_count = width / 8;

    if strip_count == 0 {
        bail!("Image width {} too small for strips", width);
    }

    let offset_table = &smap_block[8..];
    if offset_table.len() < strip_count * 4 {
        bail!("SMAP too small for offset table");
    }

    let mut pixels = vec![0u8; width * height];

    for strip_idx in 0..strip_count {
        let off = u32::from_le_bytes([
            offset_table[strip_idx * 4],
            offset_table[strip_idx * 4 + 1],
            offset_table[strip_idx * 4 + 2],
            offset_table[strip_idx * 4 + 3],
        ]) as usize;

        if off >= smap_block.len() {
            // strip offset out of bounds, skip
            continue;
        }

        let strip_data = &smap_block[off..];
        match image_decode::decode_strip(strip_data, height) {
            Ok(strip_pixels) => {
                let x_start = strip_idx * 8;
                for y in 0..height {
                    for x in 0..8 {
                        if x_start + x < width && y * 8 + x < strip_pixels.len() {
                            pixels[y * width + x_start + x] = strip_pixels[y * 8 + x];
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }

    Ok(pixels)
}

fn parse_room_header(data: &[u8], room_block: &Block, version: ScummVersion) -> Result<RoomHeader> {
    let rmhd = find_child(data, room_block, b"RMHD")
        .ok_or_else(|| anyhow::anyhow!("ROOM missing RMHD"))?;

    let d = &data[rmhd.data_offset()..rmhd.end_offset()];

    match version {
        ScummVersion::V7 => {
            // V7: version(4) + width(2) + height(2)
            if d.len() < 8 {
                bail!("V7 RMHD too small");
            }
            Ok(RoomHeader {
                width: u16::from_le_bytes([d[4], d[5]]),
                height: u16::from_le_bytes([d[6], d[7]]),
            })
        }
        _ => {
            // V5/V6: width(2) + height(2)
            if d.len() < 4 {
                bail!("RMHD too small");
            }
            Ok(RoomHeader {
                width: u16::from_le_bytes([d[0], d[1]]),
                height: u16::from_le_bytes([d[2], d[3]]),
            })
        }
    }
}

pub fn extract_palette(data: &[u8], room_block: &Block) -> Result<Palette> {
    // V5: CLUT block
    if let Some(clut) = find_child(data, room_block, b"CLUT") {
        let d = &data[clut.data_offset()..clut.end_offset()];
        if d.len() >= 768 {
            let mut colors = [(0u8, 0u8, 0u8); 256];
            for i in 0..256 {
                colors[i] = (d[i * 3], d[i * 3 + 1], d[i * 3 + 2]);
            }
            return Ok(Palette { colors });
        }
    }

    // V6/V7: PALS -> WRAP -> APAL block chain
    if let Some(pals) = find_child(data, room_block, b"PALS") {
        if let Some(wrap) = find_child(data, &pals, b"WRAP") {
            if let Some(apal) = find_child(data, &wrap, b"APAL") {
                let d = &data[apal.data_offset()..apal.end_offset()];
                if d.len() >= 768 {
                    let mut colors = [(0u8, 0u8, 0u8); 256];
                    for i in 0..256 {
                        colors[i] = (d[i * 3], d[i * 3 + 1], d[i * 3 + 2]);
                    }
                    return Ok(Palette { colors });
                }
            }
        }
    }

    // Fallback: grayscale palette
    let mut colors = [(0u8, 0u8, 0u8); 256];
    for i in 0..256 {
        colors[i] = (i as u8, i as u8, i as u8);
    }
    Ok(Palette { colors })
}
