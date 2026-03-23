use anyhow::Result;
use super::version::SciVersion;

pub type Palette = [[u8; 3]; 256];

/// Default EGA palette used by SCI0 games that don't have VGA palette resources.
pub fn default_ega_palette() -> Palette {
    let ega_colors: [(u8, u8, u8); 16] = [
        (0x00, 0x00, 0x00), // 0: black
        (0x00, 0x00, 0xAA), // 1: blue
        (0x00, 0xAA, 0x00), // 2: green
        (0x00, 0xAA, 0xAA), // 3: cyan
        (0xAA, 0x00, 0x00), // 4: red
        (0xAA, 0x00, 0xAA), // 5: magenta
        (0xAA, 0x55, 0x00), // 6: brown
        (0xAA, 0xAA, 0xAA), // 7: light gray
        (0x55, 0x55, 0x55), // 8: dark gray
        (0x55, 0x55, 0xFF), // 9: light blue
        (0x55, 0xFF, 0x55), // 10: light green
        (0x55, 0xFF, 0xFF), // 11: light cyan
        (0xFF, 0x55, 0x55), // 12: light red
        (0xFF, 0x55, 0xFF), // 13: light magenta
        (0xFF, 0xFF, 0x55), // 14: yellow
        (0xFF, 0xFF, 0xFF), // 15: white
    ];

    let mut palette = [[0u8; 3]; 256];
    for (i, &(r, g, b)) in ega_colors.iter().enumerate() {
        palette[i] = [r, g, b];
    }
    palette
}

/// Parse a SCI palette resource into a 256-color palette.
/// Based on ScummVM GfxPalette::createFromData().
pub fn parse_palette(data: &[u8], _version: SciVersion) -> Result<Palette> {
    let mut palette = [[0u8; 3]; 256];

    if data.len() < 4 {
        anyhow::bail!("Palette data too small");
    }

    // Detect palette format — based on ScummVM GfxPalette::createFromData()
    if (data[0] == 0 && data[1] == 1) ||
       (data[0] == 0 && data[1] == 0 && data.len() >= 31 &&
        u16::from_le_bytes([data[29], data[30]]) == 0) {
        // SCI0/SCI1 palette: palOffset=260, VARIABLE format (used, R, G, B), 256 colors
        return parse_palette_sci0(data);
    }

    if data.len() >= 37 {
        // SCI1.1 palette: palFormat=data[32], palOffset=37
        return parse_palette_sci1(data);
    }

    // Fallback: try to read as raw 768-byte palette (R, G, B * 256)
    if data.len() >= 768 {
        for i in 0..256 {
            palette[i][0] = data[i * 3];
            palette[i][1] = data[i * 3 + 1];
            palette[i][2] = data[i * 3 + 2];
        }
        return Ok(palette);
    }

    anyhow::bail!("Unknown palette format (size: {})", data.len());
}

/// Parse SCI0/SCI1 early palette format.
/// Based on ScummVM: palOffset=260, palFormat=VARIABLE (4 bytes: used, R, G, B), 256 colors.
fn parse_palette_sci0(data: &[u8]) -> Result<Palette> {
    let mut palette = [[0u8; 3]; 256];

    if data.len() < 260 + 256 * 4 {
        anyhow::bail!("SCI0 palette data too small ({}, need {})", data.len(), 260 + 256 * 4);
    }

    // Palette data at offset 260: 4 bytes per color (used, R, G, B)
    let pal_offset = 260;
    for i in 0..256 {
        let offset = pal_offset + i * 4;
        let _used = data[offset];
        palette[i] = [data[offset + 1], data[offset + 2], data[offset + 3]];
    }

    Ok(palette)
}

/// Parse SCI1 VGA palette format.
/// Similar to SCI0 but with different header structure.
fn parse_palette_sci1(data: &[u8]) -> Result<Palette> {
    let mut palette = [[0u8; 3]; 256];

    if data.len() < 37 {
        anyhow::bail!("SCI1 palette too small");
    }

    let color_start = data[25] as usize;
    let color_count = u16::from_le_bytes([data[29], data[30]]) as usize;
    let format = data[32];

    let palette_start = 37;
    // Format 0 = VARIABLE (4 bytes: used, R, G, B), Format 1 = CONSTANT (3 bytes: R, G, B)
    let bytes_per_entry = if format == 0 { 4 } else { 3 };

    for i in 0..color_count {
        let idx = color_start + i;
        if idx >= 256 {
            break;
        }

        let offset = palette_start + i * bytes_per_entry;
        if offset + bytes_per_entry > data.len() {
            break;
        }

        if format == 0 {
            // Variable format: used, R, G, B
            let _used = data[offset];
            palette[idx] = [data[offset + 1], data[offset + 2], data[offset + 3]];
        } else {
            // Constant format: R, G, B
            palette[idx] = [data[offset], data[offset + 1], data[offset + 2]];
        }
    }

    Ok(palette)
}

/// Parse an embedded palette from a view resource.
pub fn parse_view_palette(data: &[u8]) -> Result<Palette> {
    let mut palette = [[0u8; 3]; 256];

    if data.len() >= 256 * 4 {
        // 4 bytes per color: used, R, G, B
        for i in 0..256 {
            let offset = i * 4;
            palette[i] = [data[offset + 1], data[offset + 2], data[offset + 3]];
        }
    } else if data.len() >= 256 * 3 {
        // 3 bytes per color: R, G, B
        for i in 0..256 {
            let offset = i * 3;
            palette[i] = [data[offset], data[offset + 1], data[offset + 2]];
        }
    } else {
        anyhow::bail!("View palette data too small ({})", data.len());
    }

    Ok(palette)
}
