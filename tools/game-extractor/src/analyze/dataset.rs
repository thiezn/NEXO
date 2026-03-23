use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::OutputMode;
use crate::extractor::common::metadata::LoraEntry;
use crate::extractor::common::output::write_jsonl;

/// Scan the dataset folder for images and existing metadata.
/// Returns (list of image paths, existing metadata keyed by image field).
pub fn load_dataset(dataset_path: &Path) -> Result<(Vec<PathBuf>, HashMap<String, LoraEntry>)> {
    let images_dir = dataset_path.join("images");
    anyhow::ensure!(
        images_dir.is_dir(),
        "No images/ directory found in {}",
        dataset_path.display()
    );

    // Collect image files
    let mut images: Vec<PathBuf> = std::fs::read_dir(&images_dir)
        .context("Failed to read images directory")?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("png" | "jpg" | "jpeg" | "webp")
            )
        })
        .collect();
    images.sort();

    // Read existing metadata if present
    let metadata_path = dataset_path.join("metadata.jsonl");
    let mut existing = HashMap::new();
    if metadata_path.is_file() {
        let content =
            std::fs::read_to_string(&metadata_path).context("Failed to read metadata.jsonl")?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<LoraEntry>(line) {
                existing.insert(entry.image.clone(), entry);
            }
        }
    }

    Ok((images, existing))
}

/// Check whether an image needs labelling.
pub fn needs_labelling(entry: Option<&LoraEntry>, force: bool) -> bool {
    if force {
        return true;
    }
    match entry {
        None => true,
        Some(e) => e.text.is_empty(),
    }
}

/// Write metadata entries to the appropriate JSONL file.
pub fn write_metadata(entries: &[LoraEntry], dataset_path: &Path, mode: &OutputMode) -> Result<()> {
    let filename = match mode {
        OutputMode::Update => "metadata.jsonl",
        OutputMode::Review => "metadata_review.jsonl",
    };
    let output_path = dataset_path.join(filename);

    write_jsonl(entries, &output_path)?;
    Ok(())
}
