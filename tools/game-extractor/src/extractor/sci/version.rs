use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SciVersion {
    Sci0,
    Sci01,
    Sci1Early,
    Sci1Middle,
    Sci1Late,
    Sci11,
    Sci2,
    Sci21,
}

impl std::fmt::Display for SciVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SciVersion::Sci0 => write!(f, "SCI0"),
            SciVersion::Sci01 => write!(f, "SCI01"),
            SciVersion::Sci1Early => write!(f, "SCI1 Early"),
            SciVersion::Sci1Middle => write!(f, "SCI1 Middle"),
            SciVersion::Sci1Late => write!(f, "SCI1 Late"),
            SciVersion::Sci11 => write!(f, "SCI1.1"),
            SciVersion::Sci2 => write!(f, "SCI2"),
            SciVersion::Sci21 => write!(f, "SCI2.1"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResMapVersion {
    Sci0,
    Sci1Middle,
    Sci1Late,
    Sci11,
    Sci2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResVolVersion {
    Sci0,
    Sci1Early,
    Sci1Late,
    Sci11,
    Sci2,
}

#[derive(Debug, Clone)]
pub struct SciGameInfo {
    pub version: SciVersion,
    pub map_version: ResMapVersion,
    pub vol_version: ResVolVersion,
    pub display_name: String,
    pub game_dir: PathBuf,
    pub map_path: PathBuf,
    pub volume_paths: Vec<PathBuf>,
}

/// Detect a SCI game in the given directory.
pub fn detect_game(dir: &Path) -> Result<SciGameInfo> {
    // Try SCI2/2.1 first: RESMAP.000 + RESSCI.000 (or RESOURCE.WIN)
    let resmap = find_file_case_insensitive(dir, "resmap.000");
    if let Some(map_path) = resmap {
        // Try RESSCI.NNN first, then fall back to RESOURCE.WIN
        let mut volume_paths = find_volume_files(dir, "ressci");
        if volume_paths.is_empty() {
            // Some SCI2.1 games (e.g., LSL7) use RESOURCE.WIN as the data file
            if let Some(res_win) = find_file_case_insensitive(dir, "resource.win") {
                volume_paths = vec![res_win];
            }
        }
        if !volume_paths.is_empty() {
            let display_name = guess_display_name(dir);
            return Ok(SciGameInfo {
                version: SciVersion::Sci21,
                map_version: ResMapVersion::Sci2,
                vol_version: ResVolVersion::Sci2,
                display_name,
                game_dir: dir.to_path_buf(),
                map_path,
                volume_paths,
            });
        }
    }

    // Try SCI0/SCI1/SCI1.1: RESOURCE.MAP + RESOURCE.NNN
    let resource_map = find_file_case_insensitive(dir, "resource.map");
    if let Some(map_path) = resource_map {
        let volume_paths = find_volume_files(dir, "resource");
        if volume_paths.is_empty() {
            anyhow::bail!("Found RESOURCE.MAP but no volume files");
        }

        let map_data = std::fs::read(&map_path)
            .context("Failed to read resource map")?;

        let map_version = detect_map_version(&map_data)?;
        let vol_version = detect_vol_version(&volume_paths[0], &map_version)?;

        let sci_version = match (&map_version, &vol_version) {
            (ResMapVersion::Sci0, _) => SciVersion::Sci0,
            (ResMapVersion::Sci1Middle, _) => SciVersion::Sci1Middle,
            (ResMapVersion::Sci1Late, _) => SciVersion::Sci1Late,
            (ResMapVersion::Sci11, _) => SciVersion::Sci11,
            (ResMapVersion::Sci2, _) => SciVersion::Sci21,
        };

        let display_name = guess_display_name(dir);

        return Ok(SciGameInfo {
            version: sci_version,
            map_version,
            vol_version,
            display_name,
            game_dir: dir.to_path_buf(),
            map_path,
            volume_paths,
        });
    }

    anyhow::bail!("No SCI game found in {}", dir.display())
}

/// Detect map version from RESOURCE.MAP data.
/// Based on ScummVM's detectMapVersion().
fn detect_map_version(data: &[u8]) -> Result<ResMapVersion> {
    if data.len() < 6 {
        anyhow::bail!("Resource map too small");
    }

    // Check for SCI0/SCI01 terminator: last 6 bytes should have 0xFFFFFFFF offset
    let len = data.len();
    if len >= 6 {
        let last_offset = u32::from_le_bytes([data[len-4], data[len-3], data[len-2], data[len-1]]);
        if last_offset == 0xFFFFFFFF {
            // SCI0 or SCI1Middle format — both use 6-byte entries with 0xFFFFFFFF terminator
            // Distinguish by volume bit scheme:
            // SCI0: volume = (offset >> 26) & 0x3F (6-bit, top 6 bits)
            // SCI1Middle: volume = (offset >> 28) & 0x0F (4-bit, top 4 bits)
            // Check which scheme gives reasonable volume numbers (max ~10)
            // Collect unique volume numbers under both interpretations
            let mut vols_sci0 = std::collections::BTreeSet::new();
            let mut vols_sci1m = std::collections::BTreeSet::new();
            let mut pos = 0;
            while pos + 6 <= len {
                let offset = u32::from_le_bytes([data[pos+2], data[pos+3], data[pos+4], data[pos+5]]);
                if offset == 0xFFFFFFFF {
                    break;
                }
                vols_sci0.insert((offset >> 26) & 0x3F);
                vols_sci1m.insert((offset >> 28) & 0x0F);
                pos += 6;
            }

            // Determine which scheme gives more plausible volume numbers.
            // SCI0: 6-bit volume (top 6 bits), 26-bit offset
            // SCI1Middle: 4-bit volume (top 4 bits), 28-bit offset
            //
            // Key insight: SCI0 volumes in bits 26-27 become part of the volume
            // under SCI0 but part of the offset under SCI1Middle. So:
            // - True SCI0 with vol=1 → SCI1Middle sees vol=0 (bits 26-27 absorbed into offset)
            // - True SCI1Middle with vol=1 → SCI0 sees vol=4 (bit 28 = bit 26+2)
            //
            // Check for "gaps": SCI1Middle data misread as SCI0 gives volumes 0,4,8,12,...
            let sci0_vols: Vec<u32> = vols_sci0.iter().copied().collect();
            let has_gaps = sci0_vols.len() >= 2 && sci0_vols.windows(2).any(|w| w[1] - w[0] > 2);

            if has_gaps {
                // Gapped volumes under SCI0 = actually SCI1Middle
                return Ok(ResMapVersion::Sci1Middle);
            }
            // Sequential volumes under SCI0 = true SCI0
            return Ok(ResMapVersion::Sci0);
        }
    }

    // SCI1+ format: directory type headers (type byte + offset word)
    // Parse type entries to determine version
    let mut pos = 0;
    let mut type_entries: Vec<(u8, u16)> = Vec::new();
    while pos + 3 <= len {
        let type_byte = data[pos];
        let offset = u16::from_le_bytes([data[pos+1], data[pos+2]]);

        let res_type = type_byte & 0x1F;
        type_entries.push((type_byte, offset));
        pos += 3;

        // The terminator has type 0xFF (or type byte & 0x1F == 0x1F)
        if res_type == 0x1F || type_byte == 0xFF {
            break;
        }
    }

    if type_entries.is_empty() {
        anyhow::bail!("Failed to parse resource map directory");
    }

    // Check if any type byte has high bit set (>= 0x80) — indicates SCI1
    // SCI2 uses raw type values (0-31)
    let has_high_types = type_entries.iter().any(|(t, _)| *t >= 0x80 && (*t & 0x1F) != 0x1F);

    if !has_high_types {
        // Could be SCI2 or SCI1 with low types
        // SCI2 typically has type values < 0x20
    }

    // Use the directory offsets to calculate entry sizes
    // Between consecutive type headers, the data is resource entries
    // Entry size helps distinguish SCI1Late (6 bytes) vs SCI1.1 (5 bytes)
    if type_entries.len() >= 2 {
        let first_offset = type_entries[0].1 as usize;
        let second_offset = type_entries[1].1 as usize;

        if second_offset > first_offset && first_offset > 0 {
            let block_size = second_offset - first_offset;
            if block_size > 0 {
                if block_size % 5 == 0 && block_size % 6 != 0 {
                    return Ok(ResMapVersion::Sci11);
                }
                if block_size % 6 == 0 && block_size % 5 != 0 {
                    // Could be SCI1Late or SCI2
                    if !has_high_types {
                        return Ok(ResMapVersion::Sci2);
                    }
                    return Ok(ResMapVersion::Sci1Late);
                }
                // Divisible by both 5 and 6 (e.g., 2100 = 420*5 = 350*6)
                // Validate by parsing a few entries under each interpretation
                // and checking which gives more plausible resource numbers
                let fo = first_offset;
                let num_test = 3.min(block_size / 6); // test at most 3 entries

                // Under 6-byte: check resource numbers are reasonable
                let mut six_ok = true;
                for j in 0..num_test {
                    let ep = fo + j * 6;
                    if ep + 6 > data.len() { six_ok = false; break; }
                    let num = u16::from_le_bytes([data[ep], data[ep+1]]);
                    if num > 5000 { six_ok = false; break; }
                }

                // Under 5-byte: check that numbers after first are still reasonable
                let mut five_ok = true;
                for j in 0..num_test {
                    let ep = fo + j * 5;
                    if ep + 5 > data.len() { five_ok = false; break; }
                    let num = u16::from_le_bytes([data[ep], data[ep+1]]);
                    if num > 5000 { five_ok = false; break; }
                }

                // If both look OK, check if 5-byte gives non-sequential numbers
                // (SCI1Late entries are sequential; 5-byte misparse gives jumbled numbers)
                if five_ok && six_ok {
                    let mut nums_5: Vec<u16> = Vec::new();
                    let mut nums_6: Vec<u16> = Vec::new();
                    for j in 0..num_test.min(5) {
                        let ep5 = fo + j * 5;
                        let ep6 = fo + j * 6;
                        if ep5 + 5 <= data.len() {
                            nums_5.push(u16::from_le_bytes([data[ep5], data[ep5+1]]));
                        }
                        if ep6 + 6 <= data.len() {
                            nums_6.push(u16::from_le_bytes([data[ep6], data[ep6+1]]));
                        }
                    }
                    // Check which has more sequential/sorted entries
                    let sorted_5 = nums_5.windows(2).all(|w| w[0] <= w[1]);
                    let sorted_6 = nums_6.windows(2).all(|w| w[0] <= w[1]);
                    if sorted_6 && !sorted_5 {
                        if !has_high_types {
                            return Ok(ResMapVersion::Sci2);
                        }
                        return Ok(ResMapVersion::Sci1Late);
                    }
                }

                // Default: prefer SCI1.1 (5-byte)
                if five_ok {
                    return Ok(ResMapVersion::Sci11);
                }
                return Ok(ResMapVersion::Sci1Late);
            }
        }
    }

    // Fallback: if we have directory headers but can't determine entry size,
    // try SCI1Late as default for the directory-based format
    Ok(ResMapVersion::Sci1Late)
}

/// Detect volume version by probing the first resource header.
fn detect_vol_version(vol_path: &Path, map_version: &ResMapVersion) -> Result<ResVolVersion> {
    let data = std::fs::read(vol_path)
        .context("Failed to read volume file for version detection")?;

    if data.len() < 13 {
        anyhow::bail!("Volume file too small");
    }

    match map_version {
        ResMapVersion::Sci0 => {
            // SCI0: 8-byte header: resId(2) + packed(2) + unpacked(2) + compression(2)
            // Validate: compression should be 0-4
            let compression = u16::from_le_bytes([data[6], data[7]]);
            if compression <= 4 {
                return Ok(ResVolVersion::Sci0);
            }
            Ok(ResVolVersion::Sci1Early)
        }
        ResMapVersion::Sci1Middle => {
            // Could be SCI0 or SCI1Early volume format
            // SCI1Early: 9-byte header: type(1) + num(2) + packed(2) + unpacked(2) + comp(2)
            let compression_sci1 = u16::from_le_bytes([data[7], data[8]]);
            if compression_sci1 <= 20 {
                return Ok(ResVolVersion::Sci1Early);
            }
            Ok(ResVolVersion::Sci0)
        }
        ResMapVersion::Sci1Late => Ok(ResVolVersion::Sci1Late),
        ResMapVersion::Sci11 => Ok(ResVolVersion::Sci11),
        ResMapVersion::Sci2 => Ok(ResVolVersion::Sci2),
    }
}

/// Find a file in the directory with case-insensitive matching.
pub fn find_file_case_insensitive(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let lower = name.to_lowercase();
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().to_lowercase() == lower {
            return Some(entry.path());
        }
    }
    None
}

/// Find all volume files matching a pattern (resource.000, resource.001, etc.)
fn find_volume_files(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let lower_prefix = prefix.to_lowercase();
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            if !name.starts_with(&lower_prefix) {
                return false;
            }
            // Match resource.000-resource.999 or ressci.000-ressci.999
            let ext = name.rsplit('.').next().unwrap_or("");
            ext.len() == 3 && ext.chars().all(|c| c.is_ascii_digit())
        })
        .map(|e| e.path())
        .collect();

    paths.sort();
    paths
}

/// Guess a display name from the directory name.
fn guess_display_name(dir: &Path) -> String {
    let dir_name = dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "Unknown SCI Game".to_string());

    // Try to clean up common directory naming patterns
    // e.g., "lsl1" -> "Leisure Suit Larry 1"
    match dir_name.to_lowercase().as_str() {
        "lsl1" => "Leisure Suit Larry 1".to_string(),
        "lsl2" => "Leisure Suit Larry 2".to_string(),
        "lsl3" => "Leisure Suit Larry 3".to_string(),
        "lsl5" => "Leisure Suit Larry 5".to_string(),
        "lsl6" => "Leisure Suit Larry 6".to_string(),
        "lsl7" => "Leisure Suit Larry 7".to_string(),
        _ => {
            // Convert underscores/hyphens to spaces and title-case
            dir_name.replace('_', " ").replace('-', " ")
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}
