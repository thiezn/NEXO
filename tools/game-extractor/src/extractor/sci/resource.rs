use std::collections::HashMap;
use anyhow::{Result, Context};

use super::decompress;
use super::resource_map::{self, ResourceEntry, ResourceType};
use super::resource_volume;
use super::version::SciGameInfo;

/// Manages loading and decompressing SCI resources.
pub struct ResourceManager {
    entries: Vec<ResourceEntry>,
    /// Index from (type, number) to position in entries vec for O(1) lookup
    index: HashMap<(ResourceType, u16), usize>,
    /// Volume data keyed by volume number
    volumes: HashMap<u16, Vec<u8>>,
    game: SciGameInfo,
}

impl ResourceManager {
    pub fn new(game: &SciGameInfo) -> Result<Self> {
        let entries = resource_map::parse_resource_map(game)
            .context("Failed to parse resource map")?;

        // Build index for O(1) lookup by (type, number)
        let index: HashMap<(ResourceType, u16), usize> = entries.iter()
            .enumerate()
            .map(|(i, e)| ((e.res_type, e.number), i))
            .collect();

        // Load all volume files into memory
        let mut volumes = HashMap::new();
        for path in &game.volume_paths {
            // Extract volume number from filename extension (e.g., resource.001 -> 1)
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("000");
            let vol_num: u16 = ext.parse().unwrap_or(0);

            let data = std::fs::read(path)
                .with_context(|| format!("Failed to read volume file {}", path.display()))?;
            volumes.insert(vol_num, data);
        }

        Ok(Self {
            entries,
            index,
            volumes,
            game: game.clone(),
        })
    }

    /// Get a decompressed resource by type and number.
    pub fn get_resource(&self, res_type: ResourceType, number: u16) -> Result<Vec<u8>> {
        let entry_idx = self.index.get(&(res_type, number))
            .ok_or_else(|| anyhow::anyhow!("Resource {} {} not found", res_type.name(), number))?;
        let entry = &self.entries[*entry_idx];

        let vol_data = self.volumes.get(&entry.volume)
            .ok_or_else(|| anyhow::anyhow!("Volume {} not loaded", entry.volume))?;

        let header = resource_volume::read_resource_header(
            vol_data,
            entry.offset as usize,
            self.game.vol_version,
        ).with_context(|| format!("Failed to read header for {} {}", res_type.name(), number))?;

        let raw_data = resource_volume::read_resource_data(
            vol_data,
            entry.offset as usize,
            &header,
        ).with_context(|| format!("Failed to read data for {} {}", res_type.name(), number))?;

        if header.compression == 0 {
            Ok(raw_data.to_vec())
        } else {
            // Sanity check: unpacked_size shouldn't be unreasonably large
            if header.unpacked_size > 10_000_000 {
                anyhow::bail!("Unreasonable unpacked_size {} for {} {}",
                    header.unpacked_size, res_type.name(), number);
            }
            decompress::decompress(
                raw_data,
                header.unpacked_size,
                header.compression,
                self.game.version,
            ).with_context(|| format!("Failed to decompress {} {} (comp={})",
                res_type.name(), number, header.compression))
        }
    }

    /// List all resource numbers of a given type.
    pub fn list_resources(&self, res_type: ResourceType) -> Vec<u16> {
        let mut numbers: Vec<u16> = self.entries.iter()
            .filter(|e| e.res_type == res_type)
            .map(|e| e.number)
            .collect();
        numbers.sort();
        numbers.dedup();
        numbers
    }

    /// Get total count of resources per type (for summary display).
    pub fn resource_counts(&self) -> HashMap<&'static str, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.res_type.name()).or_insert(0) += 1;
        }
        counts
    }
}
