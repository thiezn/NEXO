mod cli;

use clap::Parser;
use cli::{Cli, Command};
use speech_to_text::config::AppConfig;

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
        Command::Transcribe {
            file,
            model,
            format,
            language,
            translate,
            output,
            no_timestamps,
        } => {
            let app_config = AppConfig::load()?;
            let model_name = model.unwrap_or(app_config.default_model.clone());

            let config = speech_to_text::TranscriptionConfig {
                model: model_name,
                language,
                translate,
                timestamps: !no_timestamps,
            };

            let path = utl_helpers::resolve_path_str(&file)?;
            let result = speech_to_text::transcribe(&config, &path, &app_config)?;

            let formatted = speech_to_text::output::format_output(&result, format.into());

            if let Some(output_path) = output {
                std::fs::write(&output_path, &formatted)?;
                tracing::info!("output written to {output_path}");
            } else {
                println!("{formatted}");
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
    use speech_to_text::manifest::{find_manifest, known_manifests, resolve_model_name};

    if model_name == "all" {
        for manifest in known_manifests() {
            tracing::info!("pulling {}...", manifest.name);
            let downloads = pull_model(manifest).await?;
            save_model_config(&manifest.name, &downloads, manifest)?;
        }
        return Ok(());
    }

    let canonical = resolve_model_name(model_name);
    let manifest =
        find_manifest(&canonical).ok_or_else(|| anyhow::anyhow!("unknown model: {canonical}"))?;

    tracing::info!("pulling {}...", manifest.name);
    let downloads = pull_model(manifest).await?;
    save_model_config(&manifest.name, &downloads, manifest)?;
    tracing::info!("done. model ready: {}", manifest.name);

    Ok(())
}

fn save_model_config(
    name: &str,
    downloads: &[(speech_to_text::manifest::WhisperComponent, std::path::PathBuf)],
    manifest: &local_inference_helpers::manifest::ModelManifest<
        speech_to_text::manifest::WhisperComponent,
    >,
) -> anyhow::Result<()> {
    use speech_to_text::config::{AppConfig, WhisperModelPaths};

    let paths = WhisperModelPaths::from_downloads(downloads)
        .ok_or_else(|| anyhow::anyhow!("missing required components after download"))?;

    let model_config = paths.to_model_config(&manifest.description);

    let mut app_config = AppConfig::load()?;
    app_config.upsert_model(name.to_string(), model_config);
    app_config.save()?;

    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    use speech_to_text::manifest::{known_manifests, total_download_size};

    let app_config = AppConfig::load()?;

    println!(
        "{:<28} {:<10} {:<8} DESCRIPTION",
        "MODEL", "FAMILY", "SIZE"
    );
    println!("{}", "-".repeat(90));

    for manifest in known_manifests() {
        let installed = app_config.models.contains_key(&manifest.name);
        let marker = if installed { " *" } else { "" };
        let size_gb = total_download_size(manifest) as f64 / 1_000_000_000.0;
        println!(
            "{:<28} {:<10} {:.1} GB  {}{}",
            manifest.name, manifest.family, size_gb, manifest.description, marker
        );
    }

    println!();
    println!("* = installed (in config)");
    println!("Default model: {}", app_config.default_model);

    Ok(())
}
