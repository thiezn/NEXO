use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScummVersion {
    V3,
    V4,
    V5,
    V6,
    V7,
}

#[derive(Debug)]
pub struct GameInfo {
    pub version: ScummVersion,
    pub xor_key: u8,
    pub index_path: PathBuf,
    pub data_path: PathBuf,
    pub base_name: String,
    pub display_name: String,
    pub sound_file: Option<PathBuf>,
    pub game_dir: PathBuf,
}

impl ScummVersion {
    pub fn xor_key(self) -> u8 {
        match self {
            ScummVersion::V3 | ScummVersion::V4 => 0xFF,
            ScummVersion::V5 | ScummVersion::V6 => 0x69,
            ScummVersion::V7 => 0x00,
        }
    }
}

fn known_display_name(stem: &str) -> Option<&'static str> {
    match stem.to_uppercase().as_str() {
        "MONKEY" => Some("The Secret of Monkey Island"),
        "MONKEY2" => Some("Monkey Island 2 LeChucks Revenge"),
        "ATLANTIS" => Some("Indiana Jones and the Fate of Atlantis"),
        "TENTACLE" => Some("Day of the Tentacle"),
        "SAMNMAX" => Some("Sam and Max Hit the Road"),
        "FT" => Some("Full Throttle"),
        _ => None,
    }
}

fn find_sound_file(dir: &Path) -> Option<PathBuf> {
    let candidates = [
        "MONSTER.SOU",
        "monster.sou",
        "Monster.sou",
        "MONSTER.SOF",
        "monster.sof",
        "Monster.sof",
        "MONSTER.SO3",
        "monster.so3",
        "MONSTER.SOG",
        "monster.sog",
    ];
    for name in &candidates {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn distinguish_v5_v6(index_path: &Path, xor_key: u8) -> Result<ScummVersion> {
    let raw = std::fs::read(index_path)?;
    if raw.len() < 16 {
        return Ok(ScummVersion::V5);
    }

    // Decrypt first 16 bytes to check the first block tag
    let mut header = [0u8; 16];
    header.copy_from_slice(&raw[..16]);
    for b in header.iter_mut() {
        *b ^= xor_key;
    }

    let tag = &header[0..4];
    if tag == b"RNAM" {
        let size = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
        // V6 has RNAM with size=9 (empty room names), V5 has larger RNAM
        if size == 9 {
            return Ok(ScummVersion::V6);
        }
    }

    // Another heuristic: check for MAXS block size differences
    // V5 MAXS = 30 bytes data, V6 MAXS = 38 bytes data
    // For now, assume V5 if RNAM > 9
    Ok(ScummVersion::V5)
}

/// Detect game files in a directory.
pub fn detect_game(dir: &Path) -> Result<GameInfo> {
    let entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();

    // Check for V7: .LA0/.LA1 pairs
    for entry in &entries {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext.to_ascii_uppercase() == "LA0" {
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                let data_path = dir.join(format!("{}.LA1", stem));
                if !data_path.exists() {
                    // Try lowercase
                    let data_path2 = dir.join(format!("{}.la1", stem));
                    if data_path2.exists() {
                        let display = known_display_name(&stem).unwrap_or(&stem).to_string();
                        return Ok(GameInfo {
                            version: ScummVersion::V7,
                            xor_key: 0x00,
                            index_path: path.clone(),
                            data_path: data_path2,
                            base_name: stem,
                            display_name: display,
                            sound_file: find_sound_file(dir),
                            game_dir: dir.to_path_buf(),
                        });
                    }
                } else {
                    let display = known_display_name(&stem).unwrap_or(&stem).to_string();
                    return Ok(GameInfo {
                        version: ScummVersion::V7,
                        xor_key: 0x00,
                        index_path: path.clone(),
                        data_path,
                        base_name: stem,
                        display_name: display,
                        sound_file: find_sound_file(dir),
                        game_dir: dir.to_path_buf(),
                    });
                }
            }
        }
    }

    // Check for V5/V6: .000/.001 pairs
    for entry in &entries {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "000" {
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                // Skip RESOURCE.000 — that's SCI, not SCUMM
                if stem.to_uppercase() == "RESOURCE" || stem.to_uppercase() == "RESSCI" {
                    continue;
                }
                // If RESOURCE.MAP exists, these .000/.001 files are SCI resource
                // volumes, not SCUMM data files (e.g. KQ4SG.000 in King's Quest 4)
                if entries.iter().any(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case("resource.map")
                }) {
                    continue;
                }
                let data_path = dir.join(format!("{}.001", stem));
                if data_path.exists() {
                    let version = distinguish_v5_v6(&path, 0x69)?;
                    let display = known_display_name(&stem).unwrap_or(&stem).to_string();
                    return Ok(GameInfo {
                        version,
                        xor_key: version.xor_key(),
                        index_path: path.clone(),
                        data_path,
                        base_name: stem,
                        display_name: display,
                        sound_file: find_sound_file(dir),
                        game_dir: dir.to_path_buf(),
                    });
                }
            }
        }
    }

    // Check for V3/V4: 00.LFL file
    let lfl_index = dir.join("00.LFL");
    if !lfl_index.exists() {
        // Try lowercase
        let lfl_index2 = dir.join("00.lfl");
        if lfl_index2.exists() {
            let dir_name = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let display = display_name_for_v3_dir(&dir_name);
            return Ok(GameInfo {
                version: ScummVersion::V3,
                xor_key: 0xFF,
                index_path: lfl_index2,
                data_path: dir.to_path_buf(),
                base_name: dir_name,
                display_name: display,
                sound_file: find_sound_file(dir),
                game_dir: dir.to_path_buf(),
            });
        }
    } else {
        let dir_name = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let display = display_name_for_v3_dir(&dir_name);
        return Ok(GameInfo {
            version: ScummVersion::V3,
            xor_key: 0xFF,
            index_path: lfl_index,
            data_path: dir.to_path_buf(),
            base_name: dir_name,
            display_name: display,
            sound_file: find_sound_file(dir),
            game_dir: dir.to_path_buf(),
        });
    }

    bail!("No SCUMM game files found in {}", dir.display())
}

fn display_name_for_v3_dir(dir_name: &str) -> String {
    let lower = dir_name.to_lowercase();
    if lower.contains("indiana") || lower.contains("indy") || lower.contains("last crusade") {
        "Indiana Jones and the Last Crusade".to_string()
    } else if lower.contains("loom") {
        "Loom".to_string()
    } else if lower.contains("zak") {
        "Zak McKracken".to_string()
    } else {
        dir_name.to_string()
    }
}
