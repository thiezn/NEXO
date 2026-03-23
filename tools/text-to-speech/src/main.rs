mod cli;

use clap::Parser;
use cli::{Cli, Command};
use text_to_speech::config::AppConfig;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);

    if let Err(e) = run(cli.command).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Generate {
            text,
            description,
            model,
            max_tokens,
            temperature,
            seed,
            output,
        } => {
            let app_config = AppConfig::load()?;
            let model_name = model.unwrap_or(app_config.default_model.clone());
            let voice_desc =
                description.unwrap_or_else(|| app_config.default_description.clone());

            let config = text_to_speech::SynthesisConfig {
                model: model_name,
                text,
                description: voice_desc,
                max_tokens,
                temperature,
                seed,
                ..Default::default()
            };

            let result = text_to_speech::synthesize(&config, &app_config)?;

            if let Some(output_path) = output {
                let path = std::path::Path::new(&output_path);
                text_to_speech::audio::wav::save_wav(
                    &result.audio.pcm_data,
                    result.audio.sample_rate,
                    path,
                )?;
                tracing::info!(
                    "Saved {:.1}s of audio to {}",
                    result.audio.duration_secs,
                    path.display()
                );
            } else {
                let wav_bytes = text_to_speech::audio::wav::encode_wav(
                    &result.audio.pcm_data,
                    result.audio.sample_rate,
                )?;
                use std::io::Write;
                std::io::stdout().write_all(&wav_bytes)?;
            }
        }

        Command::Pull { model } => {
            cmd_pull(&model).await?;
        }

        Command::List => {
            cmd_list()?;
        }
    }

    Ok(())
}

async fn cmd_pull(model_name: &str) -> anyhow::Result<()> {
    use local_inference_helpers::download::pull_model;
    use text_to_speech::manifest::{find_manifest, known_manifests, resolve_model_name};

    if model_name == "all" {
        for manifest in known_manifests() {
            tracing::info!("Pulling {}...", manifest.name);
            let downloads = pull_model(manifest).await?;
            save_model_config(&manifest.name, &downloads, manifest)?;
        }
        return Ok(());
    }

    let canonical = resolve_model_name(model_name);
    let manifest = find_manifest(&canonical)
        .ok_or_else(|| anyhow::anyhow!("Unknown model: {canonical}"))?;

    tracing::info!("Pulling {}...", manifest.name);
    let downloads = pull_model(manifest).await?;
    save_model_config(&manifest.name, &downloads, manifest)?;
    tracing::info!("Done. Model ready: {}", manifest.name);

    Ok(())
}

fn save_model_config(
    name: &str,
    downloads: &[(text_to_speech::manifest::TTSComponent, std::path::PathBuf)],
    manifest: &local_inference_helpers::manifest::ModelManifest<
        text_to_speech::manifest::TTSComponent,
    >,
) -> anyhow::Result<()> {
    use text_to_speech::config::{AppConfig, TTSModelPaths};

    let paths = TTSModelPaths::from_downloads(downloads)
        .ok_or_else(|| anyhow::anyhow!("Missing required components after download"))?;

    let model_config = paths.to_model_config(&manifest.family, &manifest.description);

    let mut app_config = AppConfig::load()?;
    app_config.upsert_model(name.to_string(), model_config);
    app_config.save()?;

    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    use text_to_speech::manifest::{known_manifests, total_download_size};

    let app_config = AppConfig::load()?;

    println!("{:<20} {:<10} {:<8} DESCRIPTION", "MODEL", "FAMILY", "SIZE");
    println!("{}", "-".repeat(80));

    for manifest in known_manifests() {
        let installed = app_config.models.contains_key(&manifest.name);
        let marker = if installed { " *" } else { "" };
        let size_gb = total_download_size(manifest) as f64 / 1_000_000_000.0;
        println!(
            "{:<20} {:<10} {:.1} GB  {}{}",
            manifest.name, manifest.family, size_gb, manifest.description, marker
        );
    }

    println!();
    println!("* = installed (in config)");
    println!("Default model: {}", app_config.default_model);

    Ok(())
}

