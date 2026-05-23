//! `models pull` command implementation.

use nexo_ai::download::manifest::Component;
use nexo_ai::download::pull::pull_model;
use nexo_ai::registry::{find_manifest, known_manifests};

/// Download one model, or all known models, into the local model directory.
///
/// # Arguments
///
/// * `model` - The manifest name to download, or `all` to fetch every known model.
/// * `force` - Whether existing downloaded files should be replaced.
///
/// # Errors
///
/// Returns an error if the model name is unknown or a download fails.
pub async fn run(model: &str, force: bool) -> cli_helpers::Result {
    let manifests: Vec<_> = if model == "all" {
        known_manifests().iter().collect()
    } else if let Some(m) = find_manifest(model) {
        vec![m]
    } else {
        let names: Vec<_> = known_manifests()
            .iter()
            .map(|m| m.manifest.name.as_str())
            .collect();
        return Err(cli_helpers::Error::Other(format!(
            "unknown model '{}'. Known models: {}",
            model,
            names.join(", ")
        )));
    };

    for entry in manifests {
        let manifest = &entry.manifest;
        println!(
            "Pulling model: {} ({:.1} GB)",
            manifest.name, manifest.size_gb
        );
        match pull_model(manifest, force).await {
            Ok(files) => {
                for (component, path) in &files {
                    println!("  {} → {}", component.name(), path.display());
                }
                println!("Done: {}", manifest.name);
            }
            Err(e) => {
                return Err(cli_helpers::Error::Other(format!(
                    "failed to pull {}: {e}",
                    manifest.name
                )));
            }
        }
    }

    Ok(())
}
