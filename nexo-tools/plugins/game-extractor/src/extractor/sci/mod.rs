pub mod decompress;
pub mod extract;
pub mod palette;
pub mod picture;
pub mod resource;
pub mod resource_map;
pub mod resource_volume;
pub mod sound;
pub mod version;
pub mod view;

use crate::extractor::common::progress::ProgressBar;
use crate::extractor::engine::{Engine, ExtractionSummary};
use anyhow::Result;
use std::path::Path;

pub struct SciEngine {
    game: version::SciGameInfo,
}

impl SciEngine {
    pub fn detect(dir: &Path) -> Option<Box<dyn Engine>> {
        version::detect_game(dir)
            .ok()
            .map(|game| Box::new(SciEngine { game }) as Box<dyn Engine>)
    }
}

impl Engine for SciEngine {
    fn name(&self) -> &str {
        "SCI"
    }
    fn game_name(&self) -> &str {
        &self.game.display_name
    }
    fn extract(
        &self,
        output_root: &Path,
        progress: Option<&ProgressBar>,
    ) -> Result<ExtractionSummary> {
        extract::extract_game(&self.game, output_root, progress)
    }
}
