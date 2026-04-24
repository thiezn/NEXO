use super::version::SciVersion;
use anyhow::Result;

/// Decompress SCI resource data based on the compression type.
/// ScummVM dispatches comp 1/2 based on getSciVersion() <= SCI_VERSION_01.
pub fn decompress(
    compressed: &[u8],
    unpacked_size: u32,
    compression: u16,
    sci_version: SciVersion,
) -> Result<Vec<u8>> {
    // SCI0 and SCI01 use: comp=1→LZW, comp=2→Huffman
    // SCI1+ use: comp=1→Huffman, comp=2→LZW1
    let is_early = matches!(sci_version, SciVersion::Sci0 | SciVersion::Sci01);

    match compression {
        0 => {
            // No compression
            Ok(compressed.to_vec())
        }
        1 => {
            if is_early {
                decompress_lzw_sci0(compressed, unpacked_size as usize)
            } else {
                decompress_huffman(compressed, unpacked_size as usize)
            }
        }
        2 => {
            if is_early {
                decompress_huffman(compressed, unpacked_size as usize)
            } else {
                decompress_lzw_sci1(compressed, unpacked_size as usize)
            }
        }
        3 => {
            // LZW1 + view reorder (ALWAYS uses SCI1-style LZW, even in SCI0 games)
            let decompressed = decompress_lzw_sci1(compressed, unpacked_size as usize)?;
            Ok(reorder_view(&decompressed))
        }
        4 => {
            // LZW1 + pic reorder (ALWAYS uses SCI1-style LZW, even in SCI0 games)
            let decompressed = decompress_lzw_sci1(compressed, unpacked_size as usize)?;
            Ok(reorder_pic(&decompressed, unpacked_size as usize))
        }
        18 | 19 | 20 => {
            // DCL (PKWARE Data Compression Library / Implode)
            decompress_dcl(compressed, unpacked_size as usize)
        }
        32 => {
            // STACpack / LZS
            decompress_lzs(compressed, unpacked_size as usize)
        }
        _ => {
            anyhow::bail!("Unknown compression type {}", compression);
        }
    }
}

// =============================================================================
// LZW Decompression (SCI0 — LSB-first, no early change)
// Matches ScummVM's DecompressorLZW::unpackLZW with kCompLZW
// =============================================================================

fn decompress_lzw_sci0(input: &[u8], unpacked_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(unpacked_size);
    let mut reader = BitReaderLsb::new(input);

    let mut code_size: u32 = 9;
    let mut table_size: u32 = 258;
    let mut code_limit: u32 = 512; // SCI0: no early change

    // Dictionary: store offset + length into output buffer (like ScummVM)
    let mut string_offsets = vec![0u32; 4096];
    let mut string_lengths = vec![0u32; 4096];

    loop {
        if output.len() >= unpacked_size {
            break;
        }

        let code = reader.read_bits(code_size)?;

        if code == 257 {
            // Terminator
            break;
        }

        if code == 256 {
            // Reset
            code_size = 9;
            table_size = 258;
            code_limit = 512;
            continue;
        }

        if code >= table_size {
            // Invalid code — exceeds current table
            break;
        }

        let new_string_offset = output.len() as u32;

        if code <= 255 {
            // Literal byte
            output.push(code as u8);
        } else {
            // Dictionary entry — copy from output buffer
            let off = string_offsets[code as usize] as usize;
            let len = string_lengths[code as usize] as usize;
            for i in 0..len {
                if output.len() >= unpacked_size {
                    break;
                }
                let byte = if off + i < output.len() {
                    output[off + i]
                } else {
                    output[off]
                };
                output.push(byte);
            }
        }

        // Add new dictionary entry (matching ScummVM order)
        if table_size < 4096 {
            // SCI0: increase code size BEFORE adding entry (ScummVM checks first)
            if table_size == code_limit && code_size < 12 {
                code_size += 1;
                code_limit = 1 << code_size;
            }

            string_offsets[table_size as usize] = new_string_offset;
            string_lengths[table_size as usize] = (output.len() as u32) - new_string_offset + 1;
            table_size += 1;
        }
    }

    output.truncate(unpacked_size);
    Ok(output)
}

// =============================================================================
// LZW Decompression (SCI01/1 — MSB-first, with "early change")
// Matches ScummVM's DecompressorLZW::unpackLZW with kCompLZW1/kCompLZW1View/kCompLZW1Pic
// =============================================================================

fn decompress_lzw_sci1(input: &[u8], unpacked_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(unpacked_size);
    let mut reader = BitReaderMsb::new(input);

    let mut code_size: u32 = 9;
    let mut table_size: u32 = 258;
    let mut code_limit: u32 = 511; // SCI1: early change (one less)

    let mut string_offsets = vec![0u32; 4096];
    let mut string_lengths = vec![0u32; 4096];

    loop {
        if output.len() >= unpacked_size {
            break;
        }

        let code = reader.read_bits(code_size)?;

        if code == 257 {
            break;
        }

        if code == 256 {
            code_size = 9;
            table_size = 258;
            code_limit = 511;
            continue;
        }

        if code >= table_size {
            break;
        }

        let new_string_offset = output.len() as u32;

        if code <= 255 {
            output.push(code as u8);
        } else {
            let off = string_offsets[code as usize] as usize;
            let len = string_lengths[code as usize] as usize;
            for i in 0..len {
                if output.len() >= unpacked_size {
                    break;
                }
                let byte = if off + i < output.len() {
                    output[off + i]
                } else {
                    output[off]
                };
                output.push(byte);
            }
        }

        if table_size < 4096 {
            // SCI1: increase code size BEFORE adding entry (ScummVM checks first)
            if table_size == code_limit && code_size < 12 {
                code_size += 1;
                code_limit = (1 << code_size) - 1;
            }

            string_offsets[table_size as usize] = new_string_offset;
            string_lengths[table_size as usize] = (output.len() as u32) - new_string_offset + 1;
            table_size += 1;
        }
    }

    output.truncate(unpacked_size);
    Ok(output)
}

// =============================================================================
// Huffman Decompression
// Matches ScummVM's DecompressorHuffman::unpack + getc2
// =============================================================================

fn decompress_huffman(input: &[u8], unpacked_size: usize) -> Result<Vec<u8>> {
    if input.len() < 2 {
        anyhow::bail!("Huffman: input too small");
    }

    let num_nodes = input[0] as usize;
    let terminator = input[1] as u16 | 0x100; // 16-bit terminator (can't match data bytes)

    if num_nodes == 0 {
        anyhow::bail!("Huffman: zero nodes");
    }

    let node_data_size = num_nodes * 2;
    if input.len() < 2 + node_data_size {
        anyhow::bail!("Huffman: not enough data for {} nodes", num_nodes);
    }

    // Node data starts at offset 2 (right after num_nodes and terminator)
    let nodes = &input[2..2 + node_data_size];
    let data_start = 2 + node_data_size;
    let mut reader = BitReaderMsb::new(&input[data_start..]);
    let mut output = Vec::with_capacity(unpacked_size);

    while output.len() < unpacked_size {
        let c = huffman_getc2(nodes, &mut reader)?;
        if c == terminator {
            break;
        }
        output.push(c as u8);
    }

    output.truncate(unpacked_size);
    Ok(output)
}

/// Traverse Huffman tree to get next decoded value.
/// Returns value (0-255 for data, 0x100+ for terminator/escape).
fn huffman_getc2(nodes: &[u8], reader: &mut BitReaderMsb) -> Result<u16> {
    let mut node_offset = 0usize; // start at first node

    loop {
        if node_offset + 1 >= nodes.len() {
            anyhow::bail!("Huffman: node offset out of bounds");
        }

        // node[1] == 0 means leaf node
        if nodes[node_offset + 1] == 0 {
            // Leaf: return node[0] | (node[1] << 8) = just node[0] since node[1] = 0
            return Ok(nodes[node_offset] as u16);
        }

        let bit = reader.read_bits(1)?;
        let next = if bit != 0 {
            // Right child: lower 4 bits of node[1]
            let n = nodes[node_offset + 1] & 0x0F;
            if n == 0 {
                // Escape: read a full byte MSB
                return Ok(reader.read_bits(8)? as u16 | 0x100);
            }
            n
        } else {
            // Left child: upper 4 bits of node[1]
            nodes[node_offset + 1] >> 4
        };

        node_offset += (next as usize) << 1;
    }
}

// =============================================================================
// DCL (Implode) Decompression
// =============================================================================

/// DCL Huffman tree node: HUFFMAN_LEAF flag in bit 30, left child in bits 23-12, right child in bits 11-0.
const HUFFMAN_LEAF: i32 = 0x40000000;

/// Traverse a DCL Huffman tree using LSB-first bit reading.
fn dcl_huffman_lookup(tree: &[i32], reader: &mut BitReaderLsb) -> Result<i32> {
    let mut pos = 0usize;
    while (tree[pos] & HUFFMAN_LEAF) == 0 {
        let bit = reader.read_bits(1)?;
        pos = if bit != 0 {
            (tree[pos] & 0xFFF) as usize
        } else {
            ((tree[pos] >> 12) & 0xFFF) as usize
        };
    }
    Ok(tree[pos] & 0xFFFF)
}

// Huffman trees from ScummVM dcl.cpp — branch nodes: (left << 12) | right; leaf nodes: value | HUFFMAN_LEAF
#[rustfmt::skip]
static DCL_LENGTH_TREE: &[i32] = &[
    (1 << 12) | 2,
    (3 << 12) | 4,     (5 << 12) | 6,
    (7 << 12) | 8,     (9 << 12) | 10,    (11 << 12) | 12,   1 | HUFFMAN_LEAF,
    (13 << 12) | 14,   (15 << 12) | 16,   (17 << 12) | 18,   3 | HUFFMAN_LEAF,   2 | HUFFMAN_LEAF,   0 | HUFFMAN_LEAF,
    (19 << 12) | 20,   (21 << 12) | 22,   (23 << 12) | 24,   6 | HUFFMAN_LEAF,   5 | HUFFMAN_LEAF,   4 | HUFFMAN_LEAF,
    (25 << 12) | 26,   (27 << 12) | 28,   10 | HUFFMAN_LEAF,  9 | HUFFMAN_LEAF,   8 | HUFFMAN_LEAF,   7 | HUFFMAN_LEAF,
    (29 << 12) | 30,   13 | HUFFMAN_LEAF,  12 | HUFFMAN_LEAF,  11 | HUFFMAN_LEAF,
    15 | HUFFMAN_LEAF,  14 | HUFFMAN_LEAF,
];

#[rustfmt::skip]
static DCL_DISTANCE_TREE: &[i32] = &[
    (1 << 12) | 2,
    (3 << 12) | 4,       (5 << 12) | 6,
    (7 << 12) | 8,       (9 << 12) | 10,      (11 << 12) | 12,     0 | HUFFMAN_LEAF,
    (13 << 12) | 14,     (15 << 12) | 16,     (17 << 12) | 18,     (19 << 12) | 20,
    (21 << 12) | 22,     (23 << 12) | 24,
    (25 << 12) | 26,     (27 << 12) | 28,     (29 << 12) | 30,     (31 << 12) | 32,
    (33 << 12) | 34,     (35 << 12) | 36,     (37 << 12) | 38,     (39 << 12) | 40,
    (41 << 12) | 42,     (43 << 12) | 44,     2 | HUFFMAN_LEAF,     1 | HUFFMAN_LEAF,
    (45 << 12) | 46,     (47 << 12) | 48,     (49 << 12) | 50,     (51 << 12) | 52,
    (53 << 12) | 54,     (55 << 12) | 56,     (57 << 12) | 58,     (59 << 12) | 60,
    (61 << 12) | 62,     (63 << 12) | 64,     (65 << 12) | 66,     (67 << 12) | 68,
    (69 << 12) | 70,     (71 << 12) | 72,     (73 << 12) | 74,     (75 << 12) | 76,
    6 | HUFFMAN_LEAF,     5 | HUFFMAN_LEAF,     4 | HUFFMAN_LEAF,     3 | HUFFMAN_LEAF,
    (77 << 12) | 78,     (79 << 12) | 80,     (81 << 12) | 82,     (83 << 12) | 84,
    (85 << 12) | 86,     (87 << 12) | 88,     (89 << 12) | 90,     (91 << 12) | 92,
    (93 << 12) | 94,     (95 << 12) | 96,     (97 << 12) | 98,     (99 << 12) | 100,
    (101 << 12) | 102,   (103 << 12) | 104,   (105 << 12) | 106,   (107 << 12) | 108,
    (109 << 12) | 110,   21 | HUFFMAN_LEAF,    20 | HUFFMAN_LEAF,    19 | HUFFMAN_LEAF,
    18 | HUFFMAN_LEAF,    17 | HUFFMAN_LEAF,    16 | HUFFMAN_LEAF,    15 | HUFFMAN_LEAF,
    14 | HUFFMAN_LEAF,    13 | HUFFMAN_LEAF,    12 | HUFFMAN_LEAF,    11 | HUFFMAN_LEAF,
    10 | HUFFMAN_LEAF,    9 | HUFFMAN_LEAF,     8 | HUFFMAN_LEAF,     7 | HUFFMAN_LEAF,
    (111 << 12) | 112,   (113 << 12) | 114,   (115 << 12) | 116,   (117 << 12) | 118,
    (119 << 12) | 120,   (121 << 12) | 122,   (123 << 12) | 124,   (125 << 12) | 126,
    47 | HUFFMAN_LEAF,    46 | HUFFMAN_LEAF,    45 | HUFFMAN_LEAF,    44 | HUFFMAN_LEAF,
    43 | HUFFMAN_LEAF,    42 | HUFFMAN_LEAF,    41 | HUFFMAN_LEAF,    40 | HUFFMAN_LEAF,
    39 | HUFFMAN_LEAF,    38 | HUFFMAN_LEAF,    37 | HUFFMAN_LEAF,    36 | HUFFMAN_LEAF,
    35 | HUFFMAN_LEAF,    34 | HUFFMAN_LEAF,    33 | HUFFMAN_LEAF,    32 | HUFFMAN_LEAF,
    31 | HUFFMAN_LEAF,    30 | HUFFMAN_LEAF,    29 | HUFFMAN_LEAF,    28 | HUFFMAN_LEAF,
    27 | HUFFMAN_LEAF,    26 | HUFFMAN_LEAF,    25 | HUFFMAN_LEAF,    24 | HUFFMAN_LEAF,
    23 | HUFFMAN_LEAF,    22 | HUFFMAN_LEAF,    63 | HUFFMAN_LEAF,    62 | HUFFMAN_LEAF,
    61 | HUFFMAN_LEAF,    60 | HUFFMAN_LEAF,    59 | HUFFMAN_LEAF,    58 | HUFFMAN_LEAF,
    57 | HUFFMAN_LEAF,    56 | HUFFMAN_LEAF,    55 | HUFFMAN_LEAF,    54 | HUFFMAN_LEAF,
    53 | HUFFMAN_LEAF,    52 | HUFFMAN_LEAF,    51 | HUFFMAN_LEAF,    50 | HUFFMAN_LEAF,
    49 | HUFFMAN_LEAF,    48 | HUFFMAN_LEAF,
];

fn decompress_dcl(input: &[u8], unpacked_size: usize) -> Result<Vec<u8>> {
    if input.len() < 2 {
        anyhow::bail!("DCL: input too small");
    }

    let mode = input[0]; // 0 = binary, 1 = ASCII
    let dict_type = input[1] as u32; // 4=1KB, 5=2KB, 6=4KB

    if mode > 1 {
        anyhow::bail!("DCL: invalid mode {}", mode);
    }
    let dict_size: usize = match dict_type {
        4 => 1024,
        5 => 2048,
        6 => 4096,
        _ => anyhow::bail!("DCL: invalid dictionary type {}", dict_type),
    };
    let dict_mask = dict_size - 1;

    let mut reader = BitReaderLsb::new(&input[2..]);
    let mut output = Vec::with_capacity(unpacked_size);
    let mut dictionary = vec![0u8; dict_size];
    let mut dict_pos: usize = 0;

    while output.len() < unpacked_size {
        let flag = match reader.read_bits(1) {
            Ok(b) => b,
            Err(_) => break,
        };

        if flag != 0 {
            // (length, distance) pair
            let len_value = dcl_huffman_lookup(DCL_LENGTH_TREE, &mut reader)? as usize;

            let token_length = if len_value < 8 {
                len_value + 2
            } else {
                let extra_bits = len_value - 7;
                8 + (1 << extra_bits) + reader.read_bits(extra_bits as u32)? as usize
            };

            if token_length == 519 {
                break; // End of stream
            }

            let dist_value = dcl_huffman_lookup(DCL_DISTANCE_TREE, &mut reader)? as usize;

            let token_offset = if token_length == 2 {
                (dist_value << 2) | reader.read_bits(2)? as usize
            } else {
                (dist_value << dict_type) | reader.read_bits(dict_type)? as usize
            } + 1;

            if token_offset > output.len() {
                anyhow::bail!(
                    "DCL: back-reference before stream start (offset={}, written={})",
                    token_offset,
                    output.len()
                );
            }

            let dict_base = (dict_pos.wrapping_sub(token_offset)) & dict_mask;
            let mut dict_src = dict_base;
            let mut dict_dst = dict_pos;

            for _ in 0..token_length {
                let b = dictionary[dict_src];
                output.push(b);
                dictionary[dict_dst] = b;
                dict_dst = (dict_dst + 1) & dict_mask;
                dict_src = (dict_src + 1) & dict_mask;
                if dict_src == dict_pos {
                    dict_src = dict_base;
                }
            }
            dict_pos = dict_dst;
        } else {
            // Literal byte
            let value = if mode == 1 {
                // ASCII mode — not implementing full ascii_tree for now, fall back to raw byte
                reader.read_bits(8)? as u8
            } else {
                reader.read_bits(8)? as u8
            };
            output.push(value);
            dictionary[dict_pos] = value;
            dict_pos = (dict_pos + 1) % dict_size;
        }
    }

    output.truncate(unpacked_size);
    Ok(output)
}

// =============================================================================
// LZS (STACpack) Decompression — for SCI2+
// =============================================================================

/// STACpack/LZS decompressor for SCI32.
/// Based on ScummVM's DecompressorLZS::unpackLZS().
fn decompress_lzs(input: &[u8], unpacked_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(unpacked_size);
    let mut reader = BitReaderMsb::new(input);

    while output.len() < unpacked_size {
        let flag = match reader.read_bits(1) {
            Ok(b) => b,
            Err(_) => break,
        };

        if flag == 0 {
            // Literal byte
            let byte = reader.read_bits(8)? as u8;
            output.push(byte);
        } else {
            // Compressed: offset/length pair
            let offset_type = reader.read_bits(1)?;

            let offset: usize;
            if offset_type == 1 {
                // 7-bit offset
                offset = reader.read_bits(7)? as usize;
                if offset == 0 {
                    break; // End marker
                }
            } else {
                // 11-bit offset
                offset = reader.read_bits(11)? as usize;
            }

            // getCompLen() — ScummVM nested switch
            let length = lzs_get_comp_len(&mut reader)?;

            if offset == 0 || offset > output.len() {
                anyhow::bail!(
                    "LZS: invalid offset {} (buffer size {})",
                    offset,
                    output.len()
                );
            }

            let src_start = output.len() - offset;
            for i in 0..length {
                let byte = output[src_start + i];
                output.push(byte);
            }
        }
    }

    output.truncate(unpacked_size);
    Ok(output)
}

/// Read compressed copy length for LZS. Matches ScummVM's DecompressorLZS::getCompLen().
fn lzs_get_comp_len(reader: &mut BitReaderMsb) -> Result<usize> {
    match reader.read_bits(2)? {
        0 => Ok(2),
        1 => Ok(3),
        2 => Ok(4),
        _ => {
            // 3: nested switch
            match reader.read_bits(2)? {
                0 => Ok(5),
                1 => Ok(6),
                2 => Ok(7),
                _ => {
                    // 3: nibble loop
                    let mut clen = 8usize;
                    loop {
                        let nibble = reader.read_bits(4)? as usize;
                        clen += nibble;
                        if nibble != 0xF {
                            break;
                        }
                    }
                    Ok(clen)
                }
            }
        }
    }
}

// =============================================================================
// View reordering (compression type 3)
// Matches ScummVM's DecompressorLZW::reorderView
// =============================================================================

const VIEW_HEADER_COLORS_8BIT: u8 = 0x80;

fn reorder_view(src: &[u8]) -> Vec<u8> {
    if src.len() < 14 {
        return src.to_vec();
    }

    let mut dest = vec![0u8; src.len()];
    let mut seeker = 0usize; // read position in src

    // Parse main header
    let cellengths_offset = read_le_u16(src, seeker) as usize + 2;
    seeker += 2;
    let loop_headers = src[seeker] as usize;
    seeker += 1;
    let lh_present = src[seeker] as usize;
    seeker += 1;
    let lh_mask = read_le_u16(src, seeker);
    seeker += 2;
    let unknown = read_le_u16(src, seeker);
    seeker += 2;
    let pal_offset = read_le_u16(src, seeker);
    seeker += 2;
    let cel_total = read_le_u16(src, seeker) as usize;
    seeker += 2;

    // Sanity checks
    if cel_total > 1000 || loop_headers > 100 || lh_present > 100 {
        return src.to_vec();
    }
    if cellengths_offset + 2 * cel_total > src.len() {
        return src.to_vec();
    }

    // Read cel lengths
    let mut cc_lengths = Vec::with_capacity(cel_total);
    for c in 0..cel_total {
        cc_lengths.push(read_le_u16(src, cellengths_offset + 2 * c) as usize);
    }

    // Write output header
    let mut writer = 0usize;
    dest[writer] = loop_headers as u8;
    writer += 1;
    dest[writer] = VIEW_HEADER_COLORS_8BIT;
    writer += 1;
    write_le_u16(&mut dest, writer, lh_mask);
    writer += 2;
    write_le_u16(&mut dest, writer, unknown);
    writer += 2;
    write_le_u16(&mut dest, writer, pal_offset);
    writer += 2;

    let lh_ptr_start = writer;
    writer += 2 * loop_headers; // Room for loop offset table

    // Read cel counts per loop
    let mut celcounts = vec![0u8; lh_present];
    if seeker + lh_present <= src.len() {
        celcounts.copy_from_slice(&src[seeker..seeker + lh_present]);
    }
    seeker += lh_present;

    let mut celindex = 0usize;
    let mut lb: u16 = 1;
    let mut w = 0usize;
    let mut lh_last: i32 = -1;
    let mut lh_ptr = lh_ptr_start;

    // Track cel positions for pixel data placement
    let mut cc_pos: Vec<usize> = vec![0; cel_total];

    for _l in 0..loop_headers {
        if (lh_mask & lb) != 0 {
            // Loop not present — reuse last
            if lh_last == -1 {
                lh_last = 0;
            }
            write_le_u16(&mut dest, lh_ptr, lh_last as u16);
            lh_ptr += 2;
        } else {
            lh_last = writer as i32;
            write_le_u16(&mut dest, lh_ptr, writer as u16);
            lh_ptr += 2;

            if w < celcounts.len() {
                let cc = celcounts[w] as usize;
                write_le_u16(&mut dest, writer, cc as u16);
                writer += 2;
                write_le_u16(&mut dest, writer, 0);
                writer += 2;

                // Build cel offset table
                let mut chptr = writer + 2 * cc;

                for c in 0..cc {
                    if celindex + c < cel_total {
                        write_le_u16(&mut dest, writer, chptr as u16);
                        writer += 2;
                        cc_pos[celindex + c] = chptr;
                        chptr += 8 + cc_lengths[celindex + c];
                    }
                }

                // Build cel headers
                for c in 0..cc {
                    if celindex + c < cel_total
                        && seeker + 7 <= src.len()
                        && writer + 8 <= dest.len()
                    {
                        dest[writer..writer + 6].copy_from_slice(&src[seeker..seeker + 6]);
                        seeker += 6;
                        let w_val = src[seeker] as u16;
                        seeker += 1;
                        write_le_u16(&mut dest, writer + 6, w_val);
                        writer += 8;
                        writer += cc_lengths[celindex + c];
                    }
                }

                celindex += cc;
                w += 1;
            }
        }

        lb <<= 1;
    }

    // Decode RLE data: figure out where pixel data begins
    let rle_start = cellengths_offset + 2 * cel_total;
    let mut pix_pos = rle_start;

    // Skip past all RLE data to find pixel data start
    for c in 0..cel_total {
        pix_pos += get_rle_size(src, pix_pos, cc_lengths[c]);
    }

    // Now decode RLE + pixel data into cel positions
    let mut rle_pos = rle_start;
    for c in 0..cel_total {
        if cc_pos[c] + 8 <= dest.len() {
            let (new_rle, new_pix) = decode_rle(
                src,
                rle_pos,
                pix_pos,
                &mut dest,
                cc_pos[c] + 8,
                cc_lengths[c],
            );
            rle_pos = new_rle;
            pix_pos = new_pix;
        }
    }

    // Copy palette if present
    if pal_offset != 0 && writer + 3 + 256 + 4 * 256 + 4 <= dest.len() {
        dest[writer] = b'P';
        dest[writer + 1] = b'A';
        dest[writer + 2] = b'L';
        writer += 3;

        for c in 0..256u16 {
            dest[writer] = c as u8;
            writer += 1;
        }

        // The missing four bytes
        let pal_src = if seeker >= 4 { seeker - 4 } else { seeker };
        let pal_data_len = (4 * 256 + 4).min(src.len().saturating_sub(pal_src));
        if writer + pal_data_len <= dest.len() {
            dest[writer..writer + pal_data_len]
                .copy_from_slice(&src[pal_src..pal_src + pal_data_len]);
        }
    }

    dest
}

/// Get the RLE data size (number of bytes consumed from RLE stream) for a cel of given decoded size
fn get_rle_size(data: &[u8], mut pos: usize, dsize: usize) -> usize {
    let start = pos;
    let mut decoded = 0usize;

    while decoded < dsize && pos < data.len() {
        let nextbyte = data[pos];
        pos += 1;
        decoded += 1;

        match nextbyte & 0xC0 {
            0x40 | 0x00 => {
                // Literal run: nextbyte pixels follow in pixel stream (not in RLE stream)
                decoded += nextbyte as usize;
            }
            0x80 => {
                // Single pixel from pixel stream
                decoded += 1;
            }
            _ => {
                // 0xC0: repeat/skip command, no extra data
            }
        }
    }

    pos - start
}

/// Decode RLE + pixel data into output buffer
/// Returns (new_rle_pos, new_pix_pos)
fn decode_rle(
    src: &[u8],
    mut rle_pos: usize,
    mut pix_pos: usize,
    dest: &mut [u8],
    mut out_pos: usize,
    size: usize,
) -> (usize, usize) {
    let mut decoded = 0usize;

    while decoded < size {
        if rle_pos >= src.len() || out_pos >= dest.len() {
            break;
        }

        let nextbyte = src[rle_pos];
        rle_pos += 1;
        dest[out_pos] = nextbyte;
        out_pos += 1;
        decoded += 1;

        match nextbyte & 0xC0 {
            0x40 | 0x00 => {
                // Literal: copy nextbyte pixels from pixel stream
                let count = nextbyte as usize;
                for _ in 0..count {
                    if pix_pos < src.len() && out_pos < dest.len() && decoded < size {
                        dest[out_pos] = src[pix_pos];
                        pix_pos += 1;
                        out_pos += 1;
                        decoded += 1;
                    }
                }
            }
            0x80 => {
                // Single pixel from pixel stream
                if pix_pos < src.len() && out_pos < dest.len() {
                    dest[out_pos] = src[pix_pos];
                    pix_pos += 1;
                    out_pos += 1;
                    decoded += 1;
                }
            }
            _ => {
                // 0xC0: repeat/skip, no extra data needed
            }
        }
    }

    (rle_pos, pix_pos)
}

// =============================================================================
// Picture reordering (compression type 4)
// Matches ScummVM's DecompressorLZW::reorderPic
// =============================================================================

const PAL_SIZE: usize = 1284;
const EXTRA_MAGIC_SIZE: usize = 15;
const PIC_OP_OPX: u8 = 0xFE;
const PIC_OPX_SET_PALETTE: u8 = 2;
const PIC_OPX_EMBEDDED_VIEW: u8 = 1;

fn reorder_pic(src: &[u8], dsize: usize) -> Vec<u8> {
    if src.len() < 14 {
        return src.to_vec();
    }

    let mut dest = vec![0u8; dsize];
    let mut seeker = 0usize;
    let mut writer = 0usize;

    // Write palette header
    dest[writer] = PIC_OP_OPX;
    writer += 1;
    dest[writer] = PIC_OPX_SET_PALETTE;
    writer += 1;

    // Palette translation map (identity)
    for i in 0..256u16 {
        if writer < dest.len() {
            dest[writer] = i as u8;
            writer += 1;
        }
    }

    // Palette stamp (4 zero bytes)
    if writer + 4 <= dest.len() {
        writer += 4;
    }

    // Read header from source
    if seeker + 6 > src.len() {
        return src.to_vec();
    }
    let view_size = read_le_u16(src, seeker) as usize;
    seeker += 2;
    let view_start = read_le_u16(src, seeker) as usize;
    seeker += 2;
    let cdata_size = read_le_u16(src, seeker) as usize;
    seeker += 2;

    // Read viewdata (7 bytes)
    let mut viewdata = [0u8; 7];
    if seeker + 7 <= src.len() {
        viewdata.copy_from_slice(&src[seeker..seeker + 7]);
    }
    seeker += 7;

    // Copy palette data (4*256 = 1024 bytes)
    let pal_avail = src.len().saturating_sub(seeker);
    let pal_bytes = (4usize * 256).min(pal_avail);
    if pal_bytes > 0 && writer + pal_bytes <= dest.len() {
        dest[writer..writer + pal_bytes].copy_from_slice(&src[seeker..seeker + pal_bytes]);
    }
    seeker += pal_bytes;
    writer += pal_bytes;

    // Copy data between palette and view start
    if view_start > PAL_SIZE + 2 {
        let extra = view_start - PAL_SIZE - 2;
        let avail = src.len().saturating_sub(seeker);
        let copy_len = extra.min(avail);
        if copy_len > 0 && writer + copy_len <= dest.len() {
            dest[writer..writer + copy_len].copy_from_slice(&src[seeker..seeker + copy_len]);
        }
        seeker += copy_len;
    }

    // Copy trailing data
    let trailing_start = view_size + view_start + EXTRA_MAGIC_SIZE;
    if dsize > trailing_start {
        let trailing_size = dsize - trailing_start;
        let avail = src.len().saturating_sub(seeker);
        let copy_len = trailing_size.min(avail);
        if copy_len > 0 && trailing_start + copy_len <= dest.len() {
            dest[trailing_start..trailing_start + copy_len]
                .copy_from_slice(&src[seeker..seeker + copy_len]);
        }
        seeker += copy_len;
    }

    // Save cdata
    let avail = src.len().saturating_sub(seeker);
    let cdata_copy_len = cdata_size.min(avail);
    let cdata: Vec<u8> = if cdata_copy_len > 0 {
        src[seeker..seeker + cdata_copy_len].to_vec()
    } else {
        Vec::new()
    };
    seeker += cdata_copy_len;

    // Write embedded view header at view_start
    writer = view_start;
    if writer + EXTRA_MAGIC_SIZE <= dest.len() {
        dest[writer] = PIC_OP_OPX;
        writer += 1;
        dest[writer] = PIC_OPX_EMBEDDED_VIEW;
        writer += 1;
        dest[writer] = 0;
        writer += 1;
        dest[writer] = 0;
        writer += 1;
        dest[writer] = 0;
        writer += 1;
        write_le_u16(&mut dest, writer, (view_size + 8) as u16);
        writer += 2;

        dest[writer..writer + 7].copy_from_slice(&viewdata);
        writer += 7;

        dest[writer] = 0;
        writer += 1;
    }

    // Decode RLE from seeker + cdata into dest at writer
    let mut rle_pos = seeker;
    let mut cdata_pos = 0usize;

    let mut decoded = 0usize;
    while decoded < view_size && writer < dest.len() {
        if rle_pos >= src.len() {
            break;
        }
        let nextbyte = src[rle_pos];
        rle_pos += 1;
        dest[writer] = nextbyte;
        writer += 1;
        decoded += 1;

        match nextbyte & 0xC0 {
            0x40 | 0x00 => {
                let count = nextbyte as usize;
                for _ in 0..count {
                    if cdata_pos < cdata.len() && writer < dest.len() && decoded < view_size {
                        dest[writer] = cdata[cdata_pos];
                        cdata_pos += 1;
                        writer += 1;
                        decoded += 1;
                    }
                }
            }
            0x80 => {
                if cdata_pos < cdata.len() && writer < dest.len() {
                    dest[writer] = cdata[cdata_pos];
                    cdata_pos += 1;
                    writer += 1;
                    decoded += 1;
                }
            }
            _ => {}
        }
    }

    dest
}

// =============================================================================
// Helper functions
// =============================================================================

fn read_le_u16(data: &[u8], offset: usize) -> u16 {
    if offset + 2 <= data.len() {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    } else {
        0
    }
}

fn write_le_u16(data: &mut [u8], offset: usize, value: u16) {
    if offset + 2 <= data.len() {
        let bytes = value.to_le_bytes();
        data[offset] = bytes[0];
        data[offset + 1] = bytes[1];
    }
}

// =============================================================================
// Bit Readers
// =============================================================================

/// LSB-first bit reader (used by SCI0 LZW, Huffman, DCL)
struct BitReaderLsb<'a> {
    data: &'a [u8],
    pos: usize,
    dwbits: u32,
    nbits: u32,
}

impl<'a> BitReaderLsb<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            dwbits: 0,
            nbits: 0,
        }
    }

    fn fetch(&mut self) {
        while self.nbits <= 24 {
            if self.pos >= self.data.len() {
                break;
            }
            self.dwbits |= (self.data[self.pos] as u32) << self.nbits;
            self.pos += 1;
            self.nbits += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> Result<u32> {
        if self.nbits < n {
            self.fetch();
        }
        if self.nbits < n {
            anyhow::bail!("BitReaderLsb: unexpected end of data");
        }
        let ret = self.dwbits & !(0xFFFFFFFFu32 << n);
        self.dwbits >>= n;
        self.nbits -= n;
        Ok(ret)
    }
}

/// MSB-first bit reader (used by SCI1 LZW, LZS)
struct BitReaderMsb<'a> {
    data: &'a [u8],
    pos: usize,
    dwbits: u32,
    nbits: u32,
}

impl<'a> BitReaderMsb<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            dwbits: 0,
            nbits: 0,
        }
    }

    fn fetch(&mut self) {
        while self.nbits <= 24 {
            if self.pos >= self.data.len() {
                break;
            }
            self.dwbits |= (self.data[self.pos] as u32) << (24 - self.nbits);
            self.pos += 1;
            self.nbits += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> Result<u32> {
        if self.nbits < n {
            self.fetch();
        }
        if self.nbits < n {
            anyhow::bail!("BitReaderMsb: unexpected end of data");
        }
        let ret = self.dwbits >> (32 - n);
        self.dwbits <<= n;
        self.nbits -= n;
        Ok(ret)
    }
}
