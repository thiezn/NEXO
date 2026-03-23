pub mod block;
pub mod costume;
pub mod image_decode;
pub mod index;
pub mod monster_sou;
pub mod resource;
pub mod room;
pub mod sound;
pub mod version;
mod extract;

use std::path::Path;
use anyhow::Result;
use crate::extractor::common::progress::ProgressBar;
use crate::extractor::engine::{Engine, ExtractionSummary};

pub struct ScummEngine {
    game: version::GameInfo,
}

impl ScummEngine {
    pub fn detect(dir: &Path) -> Option<Box<dyn Engine>> {
        version::detect_game(dir)
            .ok()
            .map(|game| Box::new(ScummEngine { game }) as Box<dyn Engine>)
    }
}

impl Engine for ScummEngine {
    fn name(&self) -> &str {
        "SCUMM"
    }

    fn game_name(&self) -> &str {
        &self.game.display_name
    }

    fn extract(&self, output_root: &Path, progress: Option<&ProgressBar>) -> Result<ExtractionSummary> {
        extract::extract_game(&self.game, output_root, progress)
    }
}
