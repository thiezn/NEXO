mod cli;

use clap::Parser;
use cli::{Cli, Command};
use text_to_img::config::AppConfig;

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
            prompt,
            model,
            width,
            height,
            steps,
            guidance,
            seed,
            num_images,
            output,
        } => {
            let app_config = AppConfig::load()?;
            let model_name = model.unwrap_or(app_config.default_model.clone());
            let model_cfg = app_config.model_config(&model_name);

            let config = text_to_img::GenerationConfig {
                model: model_name,
                prompt,
                width: width.unwrap_or_else(|| model_cfg.effective_width(&app_config)),
                height: height.unwrap_or_else(|| model_cfg.effective_height(&app_config)),
                steps,
                guidance,
                seed,
                num_images,
                ..Default::default()
            };

            let result = text_to_img::generate_images(&config).await?;

            tracing::info!("Generated {} image(s)", result.images.len());

            if let Some(output_dir) = output {
                use base64::Engine as _;
                let dir = std::path::Path::new(&output_dir);
                std::fs::create_dir_all(dir)?;
                for img in &result.images {
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(&img.base64_data)?;
                    let filename = format!(
                        "image_{}_seed_{}.{}",
                        img.index,
                        img.seed,
                        config.output_format.extension()
                    );
                    let path = dir.join(&filename);
                    std::fs::write(&path, &data)?;
                    tracing::info!("Saved: {}", path.display());
                }
            } else {
                serde_json::to_writer_pretty(std::io::stdout(), &result)?;
                println!();
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
    use text_to_img::manifest::{find_manifest, known_manifests, resolve_model_name};

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
    downloads: &[(text_to_img::manifest::ImageComponent, std::path::PathBuf)],
    manifest: &local_inference_helpers::manifest::ModelManifest<text_to_img::manifest::ImageComponent>,
) -> anyhow::Result<()> {
    use text_to_img::config::{AppConfig, ImageModelPaths};

    let paths = ImageModelPaths::from_downloads(downloads)
        .ok_or_else(|| anyhow::anyhow!("Missing required components after download"))?;

    let is_schnell = manifest.defaults.guidance == 0.0 && manifest.defaults.steps <= 4;
    let model_config = paths.to_model_config(
        &manifest.family,
        &manifest.description,
        &manifest.defaults,
        is_schnell,
    );

    let mut app_config = AppConfig::load()?;
    app_config.upsert_model(name.to_string(), model_config);
    app_config.save()?;

    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    use text_to_img::manifest::{known_manifests, total_download_size};

    let app_config = AppConfig::load()?;

    println!("{:<28} {:<10} {:<8} DESCRIPTION", "MODEL", "FAMILY", "SIZE");
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
