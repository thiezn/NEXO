use std::path::Path;
use std::time::Instant;

use crate::config::{AppConfig, ModelPaths, validate_paths};
use crate::image_preprocess::{ImagePreprocessor, preprocess_image};
use crate::inference::DescribeRequest;
use crate::inference::factory::create_engine;
use crate::models::{DescriptionConfig, DescriptionResult};
use local_inference_helpers::device::create_device;
use local_inference_helpers::dtype::gpu_dtype;

pub fn describe_image(
    config: &DescriptionConfig,
    image_path: &Path,
    app_config: &AppConfig,
) -> anyhow::Result<DescriptionResult> {
    let start = Instant::now();
    let model_name = &config.model;

    let paths = ModelPaths::resolve(model_name, app_config).ok_or_else(|| {
        anyhow::anyhow!("model '{model_name}' not found in config. Run: multimodal pull {model_name}")
    })?;
    validate_paths(&paths)?;

    let device = create_device(|info| tracing::info!("{info}"))?;
    let dtype = gpu_dtype(&device);

    let preprocessor =
        ImagePreprocessor::from_config_file(paths.preprocessor_config.as_deref())?;
    tracing::info!(path = %image_path.display(), "preprocessing image");
    let preprocessed = preprocess_image(image_path, &preprocessor, &device, dtype)?;

    let mut engine = create_engine(model_name.clone(), paths);
    engine.load(&device, dtype)?;

    let req = DescribeRequest {
        prompt: config.prompt.clone(),
        pixel_values: preprocessed.pixel_values,
        image_grid_thw: preprocessed.image_grid_thw,
        num_image_tokens: preprocessed.num_image_tokens,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_p: config.top_p,
    };

    let response = engine.describe(&req)?;

    Ok(DescriptionResult {
        text: response.text,
        model: model_name.clone(),
        prompt_used: config.prompt.clone(),
        tokens_generated: response.tokens_generated,
        inference_time_ms: start.elapsed().as_millis() as u64,
    })
}
