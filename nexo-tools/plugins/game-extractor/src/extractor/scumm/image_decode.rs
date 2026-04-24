use crate::extractor::common::bitstream::BitReader;
use anyhow::{Result, bail};

pub fn decode_strip(strip_data: &[u8], height: usize) -> Result<Vec<u8>> {
    if strip_data.is_empty() {
        bail!("Empty strip data");
    }

    let codec = strip_data[0];
    let data = &strip_data[1..];

    // Codec mapping from ScummVM gfx.h:
    // 14-18: ZIGZAG_V (BasicV, no transparency)
    // 24-28: ZIGZAG_H (BasicH, no transparency)
    // 34-38: ZIGZAG_VT (BasicV, transparency)
    // 44-48: ZIGZAG_HT (BasicH, transparency)
    // 64-68, 104-108: MAJMIN_H, RMAJMIN_H (Complex horizontal, no transparency)
    // 84-88, 124-128: MAJMIN_HT, RMAJMIN_HT (Complex horizontal, transparency)
    match codec {
        1 => decode_raw(data, height),
        14..=18 => decode_basic(data, height, codec, true, false),
        24..=28 => decode_basic(data, height, codec, false, false),
        34..=38 => decode_basic(data, height, codec, true, true),
        44..=48 => decode_basic(data, height, codec, false, true),
        64..=68 | 104..=108 => decode_complex(data, height, codec, false),
        84..=88 | 124..=128 => decode_complex(data, height, codec, true),
        _ => {
            // unknown codec, fill with black
            Ok(vec![0u8; 8 * height])
        }
    }
}

fn decode_raw(data: &[u8], height: usize) -> Result<Vec<u8>> {
    let total = 8 * height;
    if data.len() < total {
        bail!(
            "Raw strip too short: need {} bytes, have {}",
            total,
            data.len()
        );
    }
    Ok(data[..total].to_vec())
}

/// ScummVM drawStripBasicH/V (gfx.cpp:4130-4193)
/// Bit protocol:
///   !bit1: keep color
///   bit1, !bit2: absolute color read, inc = -1
///   bit1, bit2, !bit3: color += inc
///   bit1, bit2, bit3: inc = -inc; color += inc
fn decode_basic(
    data: &[u8],
    height: usize,
    codec: u8,
    vertical: bool,
    transparent: bool,
) -> Result<Vec<u8>> {
    let param_bits = (codec % 10) as u8;
    let decomp_mask = 0xFFu8 >> (8 - param_bits);

    if data.len() < 2 {
        bail!("Strip data too short");
    }

    let mut color = data[0];
    let mut reader = BitReader::new(&data[1..]);
    let mut pixels = vec![0u8; 8 * height];
    let mut inc: i8 = -1;

    let total = 8 * height;
    for i in 0..total {
        let (x, y) = if vertical {
            (i / height, i % height)
        } else {
            (i % 8, i / 8)
        };

        if !transparent || color != 0 {
            pixels[y * 8 + x] = color;
        }

        if reader.read_bit() == 0 {
        } else if reader.read_bit() == 0 {
            color = reader.read_bits(param_bits) & decomp_mask;
            inc = -1;
        } else if reader.read_bit() == 0 {
            color = color.wrapping_add(inc as u8);
        } else {
            inc = -inc;
            color = color.wrapping_add(inc as u8);
        }
    }

    Ok(pixels)
}

/// ScummVM MajMinCodec (gfx.cpp:4975-5037)
/// Uses BitReader with 16-bit initial load, repeat mode state machine.
fn decode_complex(data: &[u8], height: usize, codec: u8, transparent: bool) -> Result<Vec<u8>> {
    let param_bits = (codec % 10) as u8;

    if data.len() < 3 {
        bail!("Strip data too short for complex codec");
    }

    // MajMinCodec::setupBitReader loads 2 bytes initially (16 bits)
    let mut color = data[0];
    let mut reader = BitReader::new_preloaded(&data[1..]);
    let mut pixels = vec![0u8; 8 * height];

    let mut repeat_mode = false;
    let mut repeat_count: u16 = 0;

    for y in 0..height {
        for x in 0..8usize {
            if !transparent || color != 0 {
                pixels[y * 8 + x] = color;
            }

            if !repeat_mode {
                if reader.read_bit() != 0 {
                    if reader.read_bit() != 0 {
                        let diff = reader.read_bits(3).wrapping_sub(4);
                        if diff != 0 {
                            color = color.wrapping_add(diff);
                        } else {
                            repeat_mode = true;
                            repeat_count = reader.read_bits(8) as u16;
                            if repeat_count > 0 {
                                repeat_count -= 1;
                            }
                        }
                    } else {
                        color = reader.read_bits(param_bits);
                    }
                }
            } else {
                // ScummVM uses pre-decrement: if (--repeatCount == 0)
                repeat_count -= 1;
                if repeat_count == 0 {
                    repeat_mode = false;
                }
            }
        }
    }

    Ok(pixels)
}

// --- V3 strip decoders (column-major traversal) ---

/// Decode a V3 strip. V3 uses column-major pixel traversal and different codecs.
pub fn decode_strip_v3(strip_data: &[u8], height: usize) -> Result<Vec<u8>> {
    if strip_data.is_empty() {
        bail!("Empty V3 strip data");
    }

    let codec = strip_data[0];
    let data = &strip_data[1..];

    match codec {
        0 => Ok(vec![0u8; 8 * height]), // empty/transparent strip
        1 => decode_raw_v3(data, height),
        2 => decode_v3_codec2(data, height),
        3 => decode_v3_codec3(data, height),
        4 => decode_v3_codec4(data, height),
        7 => decode_v3_codec7(strip_data, height), // codec 7 uses full strip_data including codec byte
        10 => decode_v3_ega(data, height),
        // V3 can also use standard ZIGZAG codecs
        14..=18 => decode_basic(data, height, codec, true, false),
        24..=28 => decode_basic(data, height, codec, false, false),
        34..=38 => decode_basic(data, height, codec, true, true),
        44..=48 => decode_basic(data, height, codec, false, true),
        _ => {
            // Unknown codec, fill with black
            Ok(vec![0u8; 8 * height])
        }
    }
}

/// V3 codec 1 for old-format: raw pixels in column-major order.
fn decode_raw_v3(data: &[u8], height: usize) -> Result<Vec<u8>> {
    let total = 8 * height;
    let mut pixels = vec![0u8; total];
    let mut src = 0;

    // Column-major: iterate columns left-to-right, then rows top-to-bottom
    for x in 0..8usize {
        for y in 0..height {
            if src >= data.len() {
                return Ok(pixels);
            }
            pixels[y * 8 + x] = data[src];
            src += 1;
        }
    }
    Ok(pixels)
}

/// V3 codec 2 (unkDecode8): RLE with column-major traversal.
/// Format: (run_length, color) pairs where run = byte + 1.
fn decode_v3_codec2(data: &[u8], height: usize) -> Result<Vec<u8>> {
    let mut pixels = vec![0u8; 8 * height];
    let mut src = 0;
    let mut x = 0usize;
    let mut y = 0usize;

    while x < 8 {
        if src + 1 >= data.len() {
            break;
        }
        let run = data[src] as usize + 1;
        let color = data[src + 1];
        src += 2;

        for _ in 0..run {
            if x >= 8 {
                break;
            }
            pixels[y * 8 + x] = color;
            y += 1;
            if y >= height {
                y = 0;
                x += 1;
            }
        }
    }
    Ok(pixels)
}

/// V3 codec 3 (unkDecode9): 4-bit nibble codec with palette run selection.
/// Uses bit-level reading with column-major traversal.
fn decode_v3_codec3(data: &[u8], height: usize) -> Result<Vec<u8>> {
    let mut pixels = vec![0u8; 8 * height];
    let mut buffer: u32 = 0;
    let mut mask: u32 = 128;
    let mut src = 0;
    let mut run: u8 = 0;
    let mut x = 0usize;
    let mut y = 0usize;

    // Inline read_bit_256 and read_n_bits as closures would be complex, use helper
    macro_rules! read_bit {
        () => {{
            mask <<= 1;
            if mask == 256 {
                buffer = if src < data.len() {
                    data[src] as u32
                } else {
                    0
                };
                src += 1;
                mask = 1;
            }
            if (buffer & mask) != 0 { 1u8 } else { 0u8 }
        }};
    }

    macro_rules! read_n_bits {
        ($n:expr) => {{
            let mut c: u8 = 0;
            for b in 0..$n {
                let bit = read_bit!();
                c += bit << b;
            }
            c
        }};
    }

    while x < 8 {
        let c = read_n_bits!(4);

        match c >> 2 {
            0 => {
                let color = read_n_bits!(4);
                let count = (c & 3) as usize + 2;
                for _ in 0..count {
                    if x >= 8 {
                        return Ok(pixels);
                    }
                    pixels[y * 8 + x] = run.wrapping_mul(16).wrapping_add(color);
                    y += 1;
                    if y >= height {
                        y = 0;
                        x += 1;
                    }
                }
            }
            1 => {
                let count = (c & 3) as usize + 1;
                for _ in 0..count {
                    let color = read_n_bits!(4);
                    if x >= 8 {
                        return Ok(pixels);
                    }
                    pixels[y * 8 + x] = run.wrapping_mul(16).wrapping_add(color);
                    y += 1;
                    if y >= height {
                        y = 0;
                        x += 1;
                    }
                }
            }
            2 => {
                run = read_n_bits!(4);
            }
            _ => {}
        }
    }
    Ok(pixels)
}

/// V3 codec 4 (unkDecode10): Local palette + RLE with column-major traversal.
fn decode_v3_codec4(data: &[u8], height: usize) -> Result<Vec<u8>> {
    if data.is_empty() {
        bail!("V3 codec 4: no data");
    }

    let numcolors = data[0] as usize;
    let mut src = 1;

    let mut local_palette = [0u8; 256];
    for i in 0..numcolors {
        if src >= data.len() {
            break;
        }
        local_palette[i] = data[src];
        src += 1;
    }

    let mut pixels = vec![0u8; 8 * height];
    let mut x = 0usize;
    let mut y = 0usize;

    while x < 8 {
        if src >= data.len() {
            break;
        }
        let color = data[src];
        src += 1;

        if (color as usize) < numcolors {
            // Single pixel using local palette
            pixels[y * 8 + x] = local_palette[color as usize];
            y += 1;
            if y >= height {
                y = 0;
                x += 1;
            }
        } else {
            // RLE run
            let run = (color as usize - numcolors) + 1;
            if src >= data.len() {
                break;
            }
            let run_color = data[src];
            src += 1;

            for _ in 0..run {
                if x >= 8 {
                    break;
                }
                pixels[y * 8 + x] = run_color;
                y += 1;
                if y >= height {
                    y = 0;
                    x += 1;
                }
            }
        }
    }
    Ok(pixels)
}

/// V3 codec 7 (unkDecode11): Delta vertical with bit protocol.
/// Note: uses column-major but outer loop is columns, inner is rows (no NEXT_ROW macro).
fn decode_v3_codec7(strip_data: &[u8], height: usize) -> Result<Vec<u8>> {
    if strip_data.len() < 2 {
        bail!("V3 codec 7: strip too short");
    }

    let mut pixels = vec![0u8; 8 * height];
    let mut buffer: u32 = 0;
    let mut mask: u32 = 128;
    let mut src = 1; // skip codec byte
    let mut inc: u8 = 1;
    let mut color = strip_data[src];
    src += 1;

    macro_rules! read_bit {
        () => {{
            mask <<= 1;
            if mask == 256 {
                buffer = if src < strip_data.len() {
                    strip_data[src] as u32
                } else {
                    0
                };
                src += 1;
                mask = 1;
            }
            if (buffer & mask) != 0 { 1 } else { 0 }
        }};
    }

    macro_rules! read_n_bits {
        ($n:expr) => {{
            let mut c: u8 = 0;
            for b in 0..$n {
                let bit: u8 = if read_bit!() != 0 { 1 } else { 0 };
                c += bit << b;
            }
            c
        }};
    }

    for x in 0..8usize {
        for y in 0..height {
            pixels[y * 8 + x] = color;

            let mut i = 0;
            for _ in 0..3 {
                let bit = read_bit!();
                if bit == 0 {
                    break;
                }
                i += 1;
            }

            match i {
                1 => {
                    inc = inc.wrapping_neg(); // inc = -inc (wrapping for u8)
                    color = color.wrapping_sub(inc);
                }
                2 => {
                    color = color.wrapping_sub(inc);
                }
                3 => {
                    inc = 1;
                    color = read_n_bits!(8);
                }
                _ => {} // i == 0: keep color
            }
        }
    }
    Ok(pixels)
}

/// V3 codec 10 (drawStripEGA): EGA compression with column-major traversal.
fn decode_v3_ega(data: &[u8], height: usize) -> Result<Vec<u8>> {
    let mut pixels = vec![0u8; 8 * height];
    let mut src = 0;
    let mut x = 0usize;
    let mut y = 0usize;

    while x < 8 {
        if src >= data.len() {
            break;
        }
        let color_byte = data[src];
        src += 1;

        if color_byte & 0x80 != 0 {
            let mut run = (color_byte & 0x3F) as usize;

            if color_byte & 0x40 != 0 {
                // Dithered run: two colors packed in one byte
                if src >= data.len() {
                    break;
                }
                let color = data[src];
                src += 1;

                if run == 0 {
                    if src >= data.len() {
                        break;
                    }
                    run = data[src] as usize;
                    src += 1;
                }

                for z in 0..run {
                    if x >= 8 {
                        break;
                    }
                    pixels[y * 8 + x] = if z & 1 != 0 { color & 0xF } else { color >> 4 };
                    y += 1;
                    if y >= height {
                        y = 0;
                        x += 1;
                    }
                }
            } else {
                // Copy from previous column
                if run == 0 {
                    if src >= data.len() {
                        break;
                    }
                    run = data[src] as usize;
                    src += 1;
                }

                for _ in 0..run {
                    if x >= 8 {
                        break;
                    }
                    let prev = if x > 0 { pixels[y * 8 + (x - 1)] } else { 0 };
                    pixels[y * 8 + x] = prev;
                    y += 1;
                    if y >= height {
                        y = 0;
                        x += 1;
                    }
                }
            }
        } else {
            // Single color run
            let color = color_byte & 0xF;
            let mut run = (color_byte >> 4) as usize;

            if run == 0 {
                if src >= data.len() {
                    break;
                }
                run = data[src] as usize;
                src += 1;
            }

            for _ in 0..run {
                if x >= 8 {
                    break;
                }
                pixels[y * 8 + x] = color;
                y += 1;
                if y >= height {
                    y = 0;
                    x += 1;
                }
            }
        }
    }
    Ok(pixels)
}

/// Decode BOMP (Bitmap Object Manipulation Protocol) image data.
/// Used for some object images in V6/V7 games.
/// Format: per row, u16 LE data_size followed by RLE-encoded row data.
/// RLE: code byte, bit0=1 → run of (code>>1)+1 copies of next byte;
///      bit0=0 → copy (code>>1)+1 literal bytes.
pub fn decode_bomp(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let mut pixels = vec![0u8; width * height];
    let mut src_pos = 0;

    for y in 0..height {
        if src_pos + 2 > data.len() {
            break;
        }
        let row_size = u16::from_le_bytes([data[src_pos], data[src_pos + 1]]) as usize;
        src_pos += 2;

        let row_end = (src_pos + row_size).min(data.len());
        let mut rp = src_pos;
        let mut dx = 0;

        while rp < row_end && dx < width {
            let code = data[rp];
            rp += 1;
            let num = ((code >> 1) + 1) as usize;
            let count = num.min(width - dx);

            if code & 1 != 0 {
                // Run: fill count pixels with next byte
                if rp >= row_end {
                    break;
                }
                let color = data[rp];
                rp += 1;
                for i in 0..count {
                    pixels[y * width + dx + i] = color;
                }
            } else {
                // Literal: copy count bytes
                for i in 0..count {
                    if rp < row_end {
                        pixels[y * width + dx + i] = data[rp];
                        rp += 1;
                    }
                }
            }
            dx += num;
        }

        src_pos = (src_pos + row_size).min(data.len());
    }

    Ok(pixels)
}
