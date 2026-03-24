use std::path::Path;
use std::time::Instant;

use crate::config::{AppConfig, ModelPaths, validate_paths};
use crate::inference::engine::Qwen35Engine;
use crate::inference::TextRequest;
use crate::models::{DescriptionConfig, DescriptionResult, TextGenerationConfig, TextGenerationResult};

pub fn generate_text(
    config: &TextGenerationConfig,
    app_config: &AppConfig,
) -> anyhow::Result<TextGenerationResult> {
    let start = Instant::now();
    let model_name = &config.model;

    let paths = ModelPaths::resolve(model_name, app_config).ok_or_else(|| {
        anyhow::anyhow!("model '{model_name}' not found in config. Run: multimodal pull {model_name}")
    })?;
    validate_paths(&paths)?;

    let mut engine = Qwen35Engine::new(model_name.clone(), paths);
    engine.load()?;

    let req = TextRequest {
        prompt: config.prompt.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_p: config.top_p,
    };

    let response = engine.generate_text(&req)?;

    Ok(TextGenerationResult {
        text: response.text,
        model: model_name.clone(),
        prompt_used: config.prompt.clone(),
        tokens_generated: response.tokens_generated,
        inference_time_ms: start.elapsed().as_millis() as u64,
    })
}

pub fn describe_image(
    _config: &DescriptionConfig,
    _image_path: &Path,
    _app_config: &AppConfig,
) -> anyhow::Result<DescriptionResult> {
    anyhow::bail!("Vision inference not yet available in MLX backend. Coming in Phase 2.")
}

pub fn describe_video(
    _config: &DescriptionConfig,
    _video_path: &Path,
    _sample_fps: f64,
    _app_config: &AppConfig,
) -> anyhow::Result<DescriptionResult> {
    anyhow::bail!("Video inference not yet available in MLX backend. Coming in Phase 2.")
}
