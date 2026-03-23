use anyhow::Result;
use local_inference_helpers::progress::{ProgressCallback, ProgressReporter};

use crate::config::TTSModelPaths;
use crate::inference::{InferenceEngine, LoadStrategy, TTSRequest, TTSResponse};

/// Qwen3-TTS engine scaffold.
///
/// This pipeline will combine a Qwen3 LLM backbone with a SNAC vocoder
/// for text-to-speech synthesis. The exact model architecture and HuggingFace
/// weights are pending research — this struct provides the interface so the
/// factory and CLI are already wired up.
pub struct Qwen3TTSEngine {
    model_name: String,
    #[allow(dead_code)]
    paths: TTSModelPaths,
    #[allow(dead_code)]
    progress: ProgressReporter,
    #[allow(dead_code)]
    load_strategy: LoadStrategy,
    loaded: bool,
}

impl Qwen3TTSEngine {
    pub fn new(
        model_name: String,
        paths: TTSModelPaths,
        load_strategy: LoadStrategy,
    ) -> Self {
        Self {
            model_name,
            paths,
            progress: ProgressReporter::default(),
            load_strategy,
            loaded: false,
        }
    }
}

impl InferenceEngine for Qwen3TTSEngine {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<()> {
        anyhow::bail!(
            "Qwen3-TTS is not yet implemented. \
             The model architecture and weights are being researched. \
             Please use a Parler-TTS model instead."
        )
    }

    fn synthesize(&mut self, _req: &TTSRequest) -> Result<TTSResponse> {
        anyhow::bail!("Qwen3-TTS is not yet implemented.")
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }
}
