use super::palette::Palette;
use super::version::SciVersion;
use anyhow::Result;

/// Extracted picture (background image).
pub struct PictureResource {
    pub width: u16,
    pub height: u16,
    pub pixels: Vec<u8>,
    pub palette: Option<Palette>,
}

/// Parse a picture resource based on SCI version.
pub fn parse_picture(
    data: &[u8],
    version: SciVersion,
    global_palette: &Palette,
) -> Result<PictureResource> {
    if data.len() < 2 {
        anyhow::bail!("Picture data too small");
    }

    // SCI1.1 bitmap pictures have header_size == 0x26 at offset 0
    let header_size = u16::from_le_bytes([data[0], data[1]]) as usize;

    if header_size == 0x26 && data.len() >= 0x26 {
        return parse_picture_sci11(data, global_palette);
    }

    // SCI0/SCI1 vector pictures
    parse_picture_vector(data, version, global_palette)
}

/// Parse SCI1.1 bitmap picture.
/// Based on ScummVM GfxPicture::drawSci11Vga().
fn parse_picture_sci11(data: &[u8], global_palette: &Palette) -> Result<PictureResource> {
    if data.len() < 0x26 {
        anyhow::bail!("SCI1.1 picture data too small");
    }

    let _priority_band_count = data[3];
    let has_cel = data[4] != 0;

    let _vector_data_offset = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
    let palette_data_offset = u32::from_le_bytes([data[28], data[29], data[30], data[31]]) as usize;
    let cel_header_offset = u32::from_le_bytes([data[32], data[33], data[34], data[35]]) as usize;

    // Parse embedded palette if present
    let palette = if palette_data_offset > 0 && palette_data_offset < data.len() {
        let pal_data = &data[palette_data_offset..];
        super::palette::parse_palette(pal_data, SciVersion::Sci11).unwrap_or(*global_palette)
    } else {
        *global_palette
    };

    if !has_cel || cel_header_offset == 0 || cel_header_offset >= data.len() {
        anyhow::bail!("SCI1.1 picture has no cel data");
    }

    // Cel header is similar to view cel
    let ch = cel_header_offset;
    if ch + 32 > data.len() {
        anyhow::bail!("SCI1.1 picture cel header truncated");
    }

    let width = u16::from_le_bytes([data[ch], data[ch + 1]]);
    let height = u16::from_le_bytes([data[ch + 2], data[ch + 3]]);
    // displaceX at ch+4, displaceY at ch+6
    let clear_key = data[ch + 8];

    let rle_offset =
        u32::from_le_bytes([data[ch + 24], data[ch + 25], data[ch + 26], data[ch + 27]]) as usize;
    let literal_offset =
        u32::from_le_bytes([data[ch + 28], data[ch + 29], data[ch + 30], data[ch + 31]]) as usize;

    if width == 0 || height == 0 {
        anyhow::bail!("SCI1.1 picture has zero dimensions");
    }

    let total = width as usize * height as usize;
    let mut pixels = vec![clear_key; total];

    if rle_offset > 0 && rle_offset < data.len() {
        let rle_data = &data[rle_offset..];
        let lit_data = if literal_offset > 0 && literal_offset < data.len() {
            Some(&data[literal_offset..])
        } else {
            None
        };

        pixels = decode_pic_rle(rle_data, lit_data, width, height, clear_key);
    }

    Ok(PictureResource {
        width,
        height,
        pixels,
        palette: Some(palette),
    })
}

/// Decode RLE data for picture cels — delegates to the shared SCI1.1 RLE decoder in view.rs.
fn decode_pic_rle(
    rle_data: &[u8],
    literal_data: Option<&[u8]>,
    width: u16,
    height: u16,
    clear_key: u8,
) -> Vec<u8> {
    super::view::decode_rle_vga11(rle_data, literal_data, width, height, clear_key)
        .unwrap_or_else(|_| vec![clear_key; width as usize * height as usize])
}

/// SCI vector picture dimensions (standard VGA resolution used by all SCI0/SCI1 games).
const VECTOR_PIC_WIDTH: u16 = 320;
const VECTOR_PIC_HEIGHT: u16 = 190;

/// Parse SCI0/SCI1 vector picture by rendering drawing commands to a bitmap.
/// Based on ScummVM's GfxPicture::drawVectorData().
fn parse_picture_vector(
    data: &[u8],
    _version: SciVersion,
    global_palette: &Palette,
) -> Result<PictureResource> {
    let width: u16 = VECTOR_PIC_WIDTH;
    let height: u16 = VECTOR_PIC_HEIGHT;
    let total = width as usize * height as usize;
    let mut pixels = vec![15u8; total]; // default to white (color 15)
    let mut current_color: u8 = 0;
    let mut visual_enabled = true;
    let mut palette = *global_palette;
    let mut has_embedded_palette = false;
    let mut _pattern_code: u8 = 0;
    let mut _pattern_texture: u8 = 0;

    let mut pos = 0;

    while pos < data.len() {
        let opcode = data[pos];
        if opcode < 0xF0 {
            // Not an opcode — skip stray data bytes
            pos += 1;
            continue;
        }
        pos += 1;

        match opcode {
            0xF0 => {
                // SET_COLOR
                if pos >= data.len() {
                    break;
                }
                current_color = data[pos];
                pos += 1;
                visual_enabled = true;
            }
            0xF1 => {
                // DISABLE_VISUAL
                visual_enabled = false;
            }
            0xF2 => {
                // SET_PRIORITY
                if pos >= data.len() {
                    break;
                }
                pos += 1; // skip priority value (we don't render priority)
            }
            0xF3 => {
                // DISABLE_PRIORITY
            }
            0xF4 => {
                // SHORT_PATTERNS
                if pos >= data.len() {
                    break;
                }
                _pattern_texture = data[pos];
                pos += 1;
                // Read absolute start coords
                if pos + 3 > data.len() {
                    break;
                }
                let (mut x, mut y) = read_abs_coords(data, &mut pos);
                draw_pattern(
                    &mut pixels,
                    width,
                    height,
                    x,
                    y,
                    _pattern_code,
                    current_color,
                    visual_enabled,
                );
                // Short relative coords for subsequent points
                while pos < data.len() && data[pos] < 0xF0 {
                    if (_pattern_code & 0x20) != 0 {
                        if pos >= data.len() {
                            break;
                        }
                        _pattern_texture = data[pos];
                        pos += 1;
                    }
                    if pos >= data.len() || data[pos] >= 0xF0 {
                        break;
                    }
                    read_rel_coords_short(data, &mut pos, &mut x, &mut y);
                    draw_pattern(
                        &mut pixels,
                        width,
                        height,
                        x,
                        y,
                        _pattern_code,
                        current_color,
                        visual_enabled,
                    );
                }
            }
            0xF5 => {
                // MEDIUM_LINES
                if pos + 3 > data.len() {
                    break;
                }
                let (mut x, mut y) = read_abs_coords(data, &mut pos);
                while pos < data.len() && data[pos] < 0xF0 {
                    let old_x = x;
                    let old_y = y;
                    read_rel_coords_med(data, &mut pos, &mut x, &mut y);
                    if visual_enabled {
                        draw_line(
                            &mut pixels,
                            width,
                            height,
                            old_x,
                            old_y,
                            x,
                            y,
                            current_color,
                        );
                    }
                }
            }
            0xF6 => {
                // LONG_LINES (absolute coordinates)
                if pos + 3 > data.len() {
                    break;
                }
                let (mut x, mut y) = read_abs_coords(data, &mut pos);
                while pos + 2 < data.len() && data[pos] < 0xF0 {
                    let old_x = x;
                    let old_y = y;
                    let coords = read_abs_coords(data, &mut pos);
                    x = coords.0;
                    y = coords.1;
                    if visual_enabled {
                        draw_line(
                            &mut pixels,
                            width,
                            height,
                            old_x,
                            old_y,
                            x,
                            y,
                            current_color,
                        );
                    }
                }
            }
            0xF7 => {
                // SHORT_LINES (short relative coordinates)
                if pos + 3 > data.len() {
                    break;
                }
                let (mut x, mut y) = read_abs_coords(data, &mut pos);
                while pos < data.len() && data[pos] < 0xF0 {
                    let old_x = x;
                    let old_y = y;
                    read_rel_coords_short(data, &mut pos, &mut x, &mut y);
                    if visual_enabled {
                        draw_line(
                            &mut pixels,
                            width,
                            height,
                            old_x,
                            old_y,
                            x,
                            y,
                            current_color,
                        );
                    }
                }
            }
            0xF8 => {
                // FILL (flood fill at absolute coordinates)
                while pos + 2 < data.len() && data[pos] < 0xF0 {
                    let (fill_x, fill_y) = read_abs_coords(data, &mut pos);
                    if visual_enabled {
                        flood_fill(&mut pixels, width, height, fill_x, fill_y, current_color);
                    }
                }
            }
            0xF9 => {
                // SET_PATTERN
                if pos >= data.len() {
                    break;
                }
                _pattern_code = data[pos];
                pos += 1;
            }
            0xFA => {
                // ABSOLUTE_PATTERN
                if (_pattern_code & 0x20) != 0 {
                    if pos >= data.len() {
                        break;
                    }
                    _pattern_texture = data[pos];
                    pos += 1;
                }
                while pos + 2 < data.len() && data[pos] < 0xF0 {
                    let (px, py) = read_abs_coords(data, &mut pos);
                    draw_pattern(
                        &mut pixels,
                        width,
                        height,
                        px,
                        py,
                        _pattern_code,
                        current_color,
                        visual_enabled,
                    );
                    if (_pattern_code & 0x20) != 0 {
                        if pos >= data.len() || data[pos] >= 0xF0 {
                            break;
                        }
                        _pattern_texture = data[pos];
                        pos += 1;
                    }
                }
            }
            0xFB => {
                // SET_CONTROL
                if pos >= data.len() {
                    break;
                }
                pos += 1; // skip control value (we don't render control)
            }
            0xFC => {
                // DISABLE_CONTROL
            }
            0xFD => {
                // MEDIUM_PATTERNS
                if (_pattern_code & 0x20) != 0 {
                    if pos >= data.len() {
                        break;
                    }
                    _pattern_texture = data[pos];
                    pos += 1;
                }
                if pos + 3 > data.len() {
                    break;
                }
                let (mut x, mut y) = read_abs_coords(data, &mut pos);
                draw_pattern(
                    &mut pixels,
                    width,
                    height,
                    x,
                    y,
                    _pattern_code,
                    current_color,
                    visual_enabled,
                );
                while pos < data.len() && data[pos] < 0xF0 {
                    if (_pattern_code & 0x20) != 0 {
                        if pos >= data.len() {
                            break;
                        }
                        _pattern_texture = data[pos];
                        pos += 1;
                    }
                    if pos >= data.len() || data[pos] >= 0xF0 {
                        break;
                    }
                    read_rel_coords_med(data, &mut pos, &mut x, &mut y);
                    draw_pattern(
                        &mut pixels,
                        width,
                        height,
                        x,
                        y,
                        _pattern_code,
                        current_color,
                        visual_enabled,
                    );
                }
            }
            0xFE => {
                // Extended opcode (PIC_OP_OPX)
                if pos >= data.len() {
                    break;
                }
                let sub_opcode = data[pos];
                pos += 1;

                match sub_opcode {
                    0x00 => {
                        // SET_PALETTE_ENTRIES (VGA): skip variable-length palette entries
                        while pos < data.len() && data[pos] < 0xF0 {
                            pos += 1;
                        }
                    }
                    0x01 => {
                        // EMBEDDED_VIEW: render bitmap cel onto canvas
                        if pos + 3 > data.len() {
                            break;
                        }
                        let (view_x, view_y) = read_abs_coords(data, &mut pos);
                        if pos + 2 > data.len() {
                            break;
                        }
                        let view_size = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                        pos += 2;
                        // Cel header: width(2) + height(2) + displaceX(1) + displaceY(1) + clearKey(1) + extra(1) = 8 bytes
                        if pos + 8 > data.len() {
                            break;
                        }
                        let cel_w = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                        let cel_h = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
                        let _displace_x = data[pos + 4] as i8;
                        let _displace_y = data[pos + 5];
                        let clear_key = data[pos + 6];
                        pos += 8;
                        // Remaining data is RLE-encoded pixel data
                        let rle_len = view_size.saturating_sub(8);
                        let rle_end = pos + rle_len.min(data.len().saturating_sub(pos));
                        let rle_data = &data[pos..rle_end];
                        // Decode RLE and draw onto canvas
                        if visual_enabled && cel_w > 0 && cel_h > 0 {
                            draw_embedded_view(
                                &mut pixels,
                                width,
                                height,
                                view_x,
                                view_y,
                                cel_w,
                                cel_h,
                                clear_key,
                                rle_data,
                            );
                        }
                        pos = rle_end;
                    }
                    0x02 => {
                        // SET_PALETTE: 256-byte mapping + 4-byte stamp + 256*4 palette
                        if pos + 256 + 4 + 256 * 4 > data.len() {
                            break;
                        }
                        pos += 256; // translation map
                        pos += 4; // stamp
                        for i in 0..256 {
                            let offset = pos + i * 4;
                            let _used = data[offset];
                            palette[i] = [data[offset + 1], data[offset + 2], data[offset + 3]];
                        }
                        has_embedded_palette = true;
                        pos += 256 * 4;
                    }
                    0x03 => {
                        // PRIORITY_TABLE_EQDIST: skip 4 bytes
                        pos += 2.min(data.len().saturating_sub(pos));
                        pos += 2.min(data.len().saturating_sub(pos));
                    }
                    0x04 => {
                        // PRIORITY_TABLE_EXPLICIT: skip 14 bytes
                        pos += 14.min(data.len().saturating_sub(pos));
                    }
                    _ => {
                        while pos < data.len() && data[pos] < 0xF0 {
                            pos += 1;
                        }
                    }
                }
            }
            0xFF => {
                break;
            }
            _ => {}
        }
    }

    Ok(PictureResource {
        width,
        height,
        pixels,
        palette: if has_embedded_palette {
            Some(palette)
        } else {
            Some(*global_palette)
        },
    })
}

/// Read 3-byte absolute coordinates: byte0 has high nibbles, byte1 = X low, byte2 = Y low.
fn read_abs_coords(data: &[u8], pos: &mut usize) -> (i16, i16) {
    if *pos + 3 > data.len() {
        return (0, 0);
    }
    let byte0 = data[*pos] as u16;
    let byte1 = data[*pos + 1] as u16;
    let byte2 = data[*pos + 2] as u16;
    *pos += 3;
    let x = (byte1 + ((byte0 & 0xF0) << 4)) as i16;
    let y = (byte2 + ((byte0 & 0x0F) << 8)) as i16;
    (x, y)
}

/// Read 1-byte short relative coordinates.
fn read_rel_coords_short(data: &[u8], pos: &mut usize, x: &mut i16, y: &mut i16) {
    if *pos >= data.len() {
        return;
    }
    let byte = data[*pos];
    *pos += 1;
    let dx = ((byte >> 4) & 0x07) as i16;
    let dy = (byte & 0x07) as i16;
    if (byte & 0x80) != 0 {
        *x -= dx;
    } else {
        *x += dx;
    }
    if (byte & 0x08) != 0 {
        *y -= dy;
    } else {
        *y += dy;
    }
}

/// Read 2-byte medium relative coordinates: byte0=Y disp, byte1=X disp.
fn read_rel_coords_med(data: &[u8], pos: &mut usize, x: &mut i16, y: &mut i16) {
    if *pos + 2 > data.len() {
        return;
    }
    let byte1 = data[*pos];
    let byte2 = data[*pos + 1];
    *pos += 2;
    if (byte1 & 0x80) != 0 {
        *y -= (byte1 & 0x7F) as i16;
    } else {
        *y += byte1 as i16;
    }
    if (byte2 & 0x80) != 0 {
        *x -= (128 - (byte2 & 0x7F)) as i16;
    } else {
        *x += byte2 as i16;
    }
}

/// Draw an embedded view cel onto the picture canvas using RLE decompression.
/// RLE format: same as SCI view cel encoding (CC=top 2 bits determine operation).
fn draw_embedded_view(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    draw_x: i16,
    draw_y: i16,
    cel_w: usize,
    cel_h: usize,
    clear_key: u8,
    rle_data: &[u8],
) {
    let cw = canvas_w as usize;
    let ch = canvas_h as usize;
    let mut rle_pos = 0;

    for row in 0..cel_h {
        let mut col = 0;
        while col < cel_w && rle_pos < rle_data.len() {
            let cmd = rle_data[rle_pos];
            rle_pos += 1;

            let command = cmd & 0xC0;
            let count = (cmd & 0x3F) as usize;
            let count = if count == 0 { 64 } else { count };

            match command {
                0xC0 => {
                    // Skip (transparent)
                    col += count;
                }
                0x80 => {
                    // Fill with next byte
                    if rle_pos >= rle_data.len() {
                        break;
                    }
                    let fill = rle_data[rle_pos];
                    rle_pos += 1;
                    for _ in 0..count {
                        if col >= cel_w {
                            break;
                        }
                        if fill != clear_key {
                            let px = draw_x as i32 + col as i32;
                            let py = draw_y as i32 + row as i32;
                            if px >= 0 && px < cw as i32 && py >= 0 && py < ch as i32 {
                                pixels[py as usize * cw + px as usize] = fill;
                            }
                        }
                        col += 1;
                    }
                }
                0x00 | 0x40 => {
                    // Literal pixels (0x00 = count up to 63, 0x40 = count + 64)
                    let cnt = if command == 0x40 { count + 64 } else { count };
                    for _ in 0..cnt {
                        if rle_pos >= rle_data.len() || col >= cel_w {
                            break;
                        }
                        let pixel = rle_data[rle_pos];
                        rle_pos += 1;
                        if pixel != clear_key {
                            let px = draw_x as i32 + col as i32;
                            let py = draw_y as i32 + row as i32;
                            if px >= 0 && px < cw as i32 && py >= 0 && py < ch as i32 {
                                pixels[py as usize * cw + px as usize] = pixel;
                            }
                        }
                        col += 1;
                    }
                }
                _ => {
                    break;
                }
            }
        }
    }
}

/// Draw a simple pattern (filled circle/rect) at the given position.
fn draw_pattern(
    pixels: &mut [u8],
    width: u16,
    height: u16,
    cx: i16,
    cy: i16,
    pattern_code: u8,
    color: u8,
    visual_enabled: bool,
) {
    if !visual_enabled {
        return;
    }
    let size = (pattern_code & 0x07) as i16;
    let is_rect = (pattern_code & 0x10) != 0;
    let w = width as i16;
    let h = height as i16;

    if is_rect {
        for dy in -size..=size {
            for dx in -size..=size {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    pixels[(py as usize) * (w as usize) + px as usize] = color;
                }
            }
        }
    } else {
        // Simple circle approximation
        let r2 = (size * size) as i16;
        for dy in -size..=size {
            for dx in -size..=size {
                if dx * dx + dy * dy <= r2 {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && px < w && py >= 0 && py < h {
                        pixels[(py as usize) * (w as usize) + px as usize] = color;
                    }
                }
            }
        }
    }
}

/// Bresenham line drawing.
fn draw_line(
    pixels: &mut [u8],
    width: u16,
    height: u16,
    x0: i16,
    y0: i16,
    x1: i16,
    y1: i16,
    color: u8,
) {
    let w = width as i32;
    let h = height as i32;

    let mut x0 = x0 as i32;
    let mut y0 = y0 as i32;
    let x1 = x1 as i32;
    let y1 = y1 as i32;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            pixels[(y0 * w + x0) as usize] = color;
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Simple flood fill.
fn flood_fill(
    pixels: &mut [u8],
    width: u16,
    height: u16,
    start_x: i16,
    start_y: i16,
    fill_color: u8,
) {
    let w = width as usize;
    let h = height as usize;

    if start_x < 0 || start_y < 0 || start_x as usize >= w || start_y as usize >= h {
        return;
    }

    let target_color = pixels[start_y as usize * w + start_x as usize];
    if target_color == fill_color {
        return;
    }

    let mut stack = vec![(start_x as usize, start_y as usize)];

    while let Some((x, y)) = stack.pop() {
        if x >= w || y >= h {
            continue;
        }

        let idx = y * w + x;
        if pixels[idx] != target_color {
            continue;
        }

        pixels[idx] = fill_color;

        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
}
