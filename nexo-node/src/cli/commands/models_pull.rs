use crate::download::{Component, find_manifest, known_manifests, pull_model};

pub async fn run(model: &str, force: bool) -> utl_helpers::Result {
    let manifests: Vec<_> = if model == "all" {
        known_manifests().iter().collect()
    } else if let Some(m) = find_manifest(model) {
        vec![m]
    } else {
        let names: Vec<_> = known_manifests()
            .iter()
            .map(|m| m.manifest.name.as_str())
            .collect();
        return Err(utl_helpers::Error::Other(format!(
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
                return Err(utl_helpers::Error::Other(format!(
                    "failed to pull {}: {e}",
                    manifest.name
                )));
            }
        }
    }

    Ok(())
}
