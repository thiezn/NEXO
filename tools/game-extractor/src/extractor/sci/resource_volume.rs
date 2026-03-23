use anyhow::Result;
use super::version::ResVolVersion;

/// Header information read from a resource in a volume file.
#[derive(Debug)]
pub struct ResourceHeader {
    pub res_type: u8,
    pub number: u16,
    pub packed_size: u32,
    pub unpacked_size: u32,
    pub compression: u16,
    pub header_size: usize,
}

/// Read a resource header from volume data at the given offset.
pub fn read_resource_header(data: &[u8], offset: usize, vol_version: ResVolVersion) -> Result<ResourceHeader> {
    let remaining = data.len().saturating_sub(offset);

    match vol_version {
        ResVolVersion::Sci0 => {
            // 8-byte header: resId(2) + packed_size(2) + unpacked_size(2) + compression(2)
            if remaining < 8 {
                anyhow::bail!("Not enough data for SCI0 resource header at offset {}", offset);
            }
            let d = &data[offset..];
            let res_id = u16::from_le_bytes([d[0], d[1]]);
            let raw_packed = u16::from_le_bytes([d[2], d[3]]) as u32;
            let unpacked_size = u16::from_le_bytes([d[4], d[5]]) as u32;
            let compression = u16::from_le_bytes([d[6], d[7]]);

            let res_type = (res_id >> 11) as u8;
            let number = res_id & 0x7FF;

            // packed_size in SCI0 ALWAYS includes 4 extra bytes
            // ScummVM: szPacked = file->readUint16LE() - 4
            let packed_size = raw_packed.saturating_sub(4);

            Ok(ResourceHeader {
                res_type,
                number,
                packed_size,
                unpacked_size,
                compression,
                header_size: 8,
            })
        }
        ResVolVersion::Sci1Early => {
            // 9-byte header: type(1) + number(2) + packed_size(2) + unpacked_size(2) + compression(2)
            // SCI1Early packed_size also includes 4 extra bytes
            if remaining < 9 {
                anyhow::bail!("Not enough data for SCI1 resource header at offset {}", offset);
            }
            let d = &data[offset..];
            let res_type = d[0];
            let number = u16::from_le_bytes([d[1], d[2]]);
            let raw_packed = u16::from_le_bytes([d[3], d[4]]) as u32;
            let unpacked_size = u16::from_le_bytes([d[5], d[6]]) as u32;
            let compression = u16::from_le_bytes([d[7], d[8]]);

            // SCI1Early also always subtracts 4
            let packed_size = raw_packed.saturating_sub(4);

            Ok(ResourceHeader {
                res_type,
                number,
                packed_size,
                unpacked_size,
                compression,
                header_size: 9,
            })
        }
        ResVolVersion::Sci1Late => {
            // 9-byte header: type(1) + number(2) + packed_size(2) + unpacked_size(2) + compression(2)
            // SCI1Late also always subtracts 4 from packed_size
            if remaining < 9 {
                anyhow::bail!("Not enough data for SCI1Late resource header at offset {}", offset);
            }
            let d = &data[offset..];
            let res_type = d[0];
            let number = u16::from_le_bytes([d[1], d[2]]);
            let raw_packed = u16::from_le_bytes([d[3], d[4]]) as u32;
            let unpacked_size = u16::from_le_bytes([d[5], d[6]]) as u32;
            let compression = u16::from_le_bytes([d[7], d[8]]);
            let packed_size = raw_packed.saturating_sub(4);

            Ok(ResourceHeader {
                res_type,
                number,
                packed_size,
                unpacked_size,
                compression,
                header_size: 9,
            })
        }
        ResVolVersion::Sci11 => {
            // 9-byte header: type(1) + number(2) + packed_size(2) + unpacked_size(2) + compression(2)
            // SCI1.1 does NOT subtract 4
            if remaining < 9 {
                anyhow::bail!("Not enough data for SCI1.1 resource header at offset {}", offset);
            }
            let d = &data[offset..];
            let res_type = d[0];
            let number = u16::from_le_bytes([d[1], d[2]]);
            let packed_size = u16::from_le_bytes([d[3], d[4]]) as u32;
            let unpacked_size = u16::from_le_bytes([d[5], d[6]]) as u32;
            let compression = u16::from_le_bytes([d[7], d[8]]);

            Ok(ResourceHeader {
                res_type,
                number,
                packed_size,
                unpacked_size,
                compression,
                header_size: 9,
            })
        }
        ResVolVersion::Sci2 => {
            // 13-byte header: type(1) + number(2) + packed_size(4) + unpacked_size(4) + compression(2)
            // For SCI2.1/SCI3: compression field is bogus; use packed != unpacked ? 32 : 0
            // No -4 subtraction on packed_size for SCI2+
            if remaining < 13 {
                anyhow::bail!("Not enough data for SCI2 resource header at offset {}", offset);
            }
            let d = &data[offset..];
            let res_type = d[0];
            let number = u16::from_le_bytes([d[1], d[2]]);
            let packed_size = u32::from_le_bytes([d[3], d[4], d[5], d[6]]);
            let unpacked_size = u32::from_le_bytes([d[7], d[8], d[9], d[10]]);
            // Compression field at bytes 11-12 is bogus for SCI2.1/SCI3
            let compression = if packed_size != unpacked_size { 32 } else { 0 };

            Ok(ResourceHeader {
                res_type,
                number,
                packed_size,
                unpacked_size,
                compression,
                header_size: 13,
            })
        }
    }
}

/// Extract raw (possibly compressed) resource data from a volume file.
/// Returns the bytes after the resource header, of length packed_size.
pub fn read_resource_data<'a>(data: &'a [u8], offset: usize, header: &ResourceHeader) -> Result<&'a [u8]> {
    let data_start = offset + header.header_size;
    let data_end = data_start + header.packed_size as usize;

    if data_end > data.len() {
        anyhow::bail!("Resource data extends beyond volume file (offset {}, packed_size {})",
            offset, header.packed_size);
    }

    Ok(&data[data_start..data_end])
}
