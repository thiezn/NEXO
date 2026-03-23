use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct Block {
    pub tag: [u8; 4],
    pub size: u32,
    /// Offset into the buffer where this block starts (at the tag).
    pub offset: usize,
}

impl Block {
    /// The offset where the block's data begins (after 8-byte header).
    pub fn data_offset(&self) -> usize {
        self.offset + 8
    }

    /// The offset where this block ends.
    pub fn end_offset(&self) -> usize {
        self.offset + self.size as usize
    }

    pub fn tag_str(&self) -> &str {
        std::str::from_utf8(&self.tag).unwrap_or("????")
    }
}

/// Parse a single block header at the given offset.
pub fn parse_block(data: &[u8], offset: usize) -> Result<Block> {
    if offset + 8 > data.len() {
        bail!("Not enough data for block header at offset {}", offset);
    }
    let tag: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    let size = u32::from_be_bytes(data[offset + 4..offset + 8].try_into().unwrap());

    if size < 8 {
        bail!(
            "Invalid block size {} for tag '{}' at offset {}",
            size,
            std::str::from_utf8(&tag).unwrap_or("????"),
            offset
        );
    }

    Ok(Block { tag, size, offset })
}

/// Iterate over child blocks within a parent block's data area.
pub fn iter_children(data: &[u8], parent: &Block) -> Vec<Block> {
    let mut children = Vec::new();
    let mut pos = parent.data_offset();
    let end = parent.end_offset();

    while pos + 8 <= end {
        match parse_block(data, pos) {
            Ok(block) => {
                let block_end = block.end_offset();
                children.push(block);
                pos = block_end;
            }
            Err(_) => break,
        }
    }

    children
}

/// Find the first child block with the given tag.
pub fn find_child(data: &[u8], parent: &Block, tag: &[u8; 4]) -> Option<Block> {
    let mut pos = parent.data_offset();
    let end = parent.end_offset();

    while pos + 8 <= end {
        match parse_block(data, pos) {
            Ok(block) => {
                if &block.tag == tag {
                    return Some(block);
                }
                pos = block.end_offset();
            }
            Err(_) => break,
        }
    }

    None
}

/// Find all child blocks with the given tag.
pub fn find_all_children(data: &[u8], parent: &Block, tag: &[u8; 4]) -> Vec<Block> {
    iter_children(data, parent)
        .into_iter()
        .filter(|b| &b.tag == tag)
        .collect()
}

/// Find a child block matching a tag prefix (e.g., b"IM" matches IM00, IM01, etc.)
pub fn find_child_with_prefix(data: &[u8], parent: &Block, prefix: &[u8]) -> Option<Block> {
    let mut pos = parent.data_offset();
    let end = parent.end_offset();

    while pos + 8 <= end {
        match parse_block(data, pos) {
            Ok(block) => {
                if block.tag.starts_with(prefix) {
                    return Some(block);
                }
                pos = block.end_offset();
            }
            Err(_) => break,
        }
    }

    None
}

/// Find all child blocks matching a tag prefix.
pub fn find_all_with_prefix(data: &[u8], parent: &Block, prefix: &[u8]) -> Vec<Block> {
    iter_children(data, parent)
        .into_iter()
        .filter(|b| b.tag.starts_with(prefix))
        .collect()
}

// --- V3/V4 "small header" block format: u32 LE size (includes 6-byte header) + 2-byte ASCII tag ---

impl Block {
    /// V3 data offset: after 6-byte header (4 size + 2 tag).
    pub fn data_offset_v3(&self) -> usize {
        self.offset + 6
    }

    /// Get the 2-byte V3 tag (first 2 bytes of the 4-byte tag field).
    pub fn tag_v3(&self) -> &[u8; 2] {
        self.tag[..2].try_into().unwrap()
    }
}

/// Parse a single V3/V4 small-header block at the given offset.
/// Format: u32 LE size (includes 6-byte header), then 2-byte ASCII tag.
pub fn parse_block_v3(data: &[u8], offset: usize) -> Result<Block> {
    if offset + 6 > data.len() {
        bail!("Not enough data for V3 block header at offset {}", offset);
    }
    let size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    let mut tag = [0u8; 4];
    tag[0] = data[offset + 4];
    tag[1] = data[offset + 5];

    if size < 6 {
        bail!("Invalid V3 block size {} at offset {}", size, offset);
    }

    Ok(Block { tag, size, offset })
}

/// Iterate over V3 child blocks within a byte range.
pub fn iter_children_v3(data: &[u8], start: usize, end: usize) -> Vec<Block> {
    let mut children = Vec::new();
    let mut pos = start;

    while pos + 6 <= end {
        match parse_block_v3(data, pos) {
            Ok(block) => {
                let block_end = block.end_offset();
                if block_end > end || block_end <= pos {
                    break;
                }
                children.push(block);
                pos = block_end;
            }
            Err(_) => break,
        }
    }

    children
}

/// Find a V3 child block by 2-byte tag within a byte range.
pub fn find_child_v3(data: &[u8], start: usize, end: usize, tag: &[u8; 2]) -> Option<Block> {
    iter_children_v3(data, start, end)
        .into_iter()
        .find(|b| b.tag_v3() == tag)
}
