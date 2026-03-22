use crate::config::{AppConfig, TTSModelPaths};
use crate::inference::factory::create_engine;
use crate::inference::{LoadStrategy, TTSRequest};
use crate::models::{GeneratedAudio, SynthesisConfig, SynthesisResult};

/// Synthesize speech from text using local inference.
pub fn synthesize(config: &SynthesisConfig, app_config: &AppConfig) -> anyhow::Result<SynthesisResult> {
    let model_name = &config.model;
    let model_cfg = app_config.model_config(model_name);

    let paths = TTSModelPaths::resolve(model_name, app_config).ok_or_else(|| {
        anyhow::anyhow!(
            "Model '{}' not configured. Run `text_to_speech pull {}` first.",
            model_name,
            model_name
        )
    })?;

    crate::config::validate_paths(&paths)?;

    let max_tokens = config
        .max_tokens
        .unwrap_or_else(|| model_cfg.effective_max_tokens(app_config));
    let temperature = config
        .temperature
        .unwrap_or_else(|| model_cfg.effective_temperature(app_config));
    let seed = config
        .seed
        .unwrap_or_else(local_inference_helpers::noise::rand_seed);

    let mut engine = create_engine(model_name, paths, app_config, LoadStrategy::Sequential)?;
    engine.load()?;

    let req = TTSRequest {
        text: config.text.clone(),
        description: config.description.clone(),
        max_tokens,
        temperature,
        seed,
    };

    let response = engine.synthesize(&req)?;

    let duration_secs = response.pcm_samples.len() as f64 / response.sample_rate as f64;

    Ok(SynthesisResult {
        text_used: req.text,
        description_used: req.description,
        audio: GeneratedAudio {
            pcm_data: response.pcm_samples,
            sample_rate: response.sample_rate,
            duration_secs,
            seed: response.seed_used,
            generation_time_ms: response.generation_time_ms,
        },
    })
}
