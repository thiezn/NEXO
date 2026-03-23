use std::path::Path;
use anyhow::Result;
use crate::extractor::common::progress::ProgressBar;

/// Summary of what was extracted from a single game.
pub struct ExtractionSummary {
    pub game_name: String,
    pub log_lines: Vec<String>,
    pub backgrounds: usize,
    pub objects: usize,
    pub sounds: usize,
    pub sprites: usize,
    pub speech_files: usize,
}

pub trait Engine: Send + Sync {
    /// Human-readable engine name (e.g., "SCUMM", "SCI")
    fn name(&self) -> &str;

    /// Display name of the detected game
    fn game_name(&self) -> &str;

    /// Run full extraction to the given output root directory.
    fn extract(&self, output_root: &Path, progress: Option<&ProgressBar>) -> Result<ExtractionSummary>;
}
