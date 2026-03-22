use base64::Engine as _;

use crate::config::{AppConfig, ImageModelPaths};
use crate::inference::factory::create_engine;
use crate::inference::{GenerateRequest, LoadStrategy};
use crate::models::{GeneratedImage, GenerationConfig, GenerationResult};

/// Generate images from a text prompt using local inference.
pub async fn generate_images(config: &GenerationConfig) -> anyhow::Result<GenerationResult> {
    let app_config = AppConfig::load()?;
    let model_name = &config.model;
    let model_cfg = app_config.model_config(model_name);

    let paths = ImageModelPaths::resolve(model_name, &app_config)
        .ok_or_else(|| anyhow::anyhow!(
            "Model '{}' not configured. Run `text_to_img pull {}` first.",
            model_name, model_name
        ))?;

    crate::config::validate_paths(&paths)?;

    let steps = config.steps.unwrap_or_else(|| model_cfg.effective_steps(&app_config));
    let guidance = config.guidance.unwrap_or_else(|| model_cfg.effective_guidance());
    let seed = config.seed.unwrap_or_else(local_inference_helpers::noise::rand_seed);

    let mut engine = create_engine(
        model_name.clone(),
        paths,
        &app_config,
        LoadStrategy::Sequential,
    )?;

    engine.load()?;

    let mut images = Vec::new();
    for i in 0..config.num_images {
        let req = GenerateRequest {
            prompt: config.prompt.clone(),
            width: config.width,
            height: config.height,
            steps,
            guidance,
            seed: seed + i as u64,
            batch_size: 1,
            output_format: config.output_format,
        };

        let response = engine.generate(&req)?;
        for img_data in response.images {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&img_data.data);
            images.push(GeneratedImage {
                index: img_data.index + i,
                base64_data: b64,
                seed: response.seed_used,
            });
        }
    }

    Ok(GenerationResult {
        prompt_used: config.prompt.clone(),
        images,
    })
}
