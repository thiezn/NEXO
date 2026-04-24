use crate::api::types::ModelCategory;
use crate::config::AiConfig;
use crate::download::pull_model;
use crate::registry::{find_manifest, known_manifests, manifests_for_category};
use anyhow::Result;

pub async fn run(model: &str, force: bool) -> Result<()> {
    // If model is "all", pull all known manifests.
    // If model matches a category name (chat, tool, etc.), pull all for that category.
    // Otherwise, try to find manifest by name.
    let manifests_to_pull: Vec<_> = if model == "all" {
        known_manifests().iter().map(|m| &m.manifest).collect()
    } else if let Ok(category) = model.parse::<ModelCategory>() {
        manifests_for_category(category)
            .iter()
            .map(|m| &m.manifest)
            .collect()
    } else if let Some(m) = find_manifest(model) {
        vec![&m.manifest]
    } else {
        anyhow::bail!(
            "unknown model or category: '{}'. Use 'nexo-ai list' to see available models.",
            model
        );
    };

    if manifests_to_pull.is_empty() {
        println!("no models found for '{model}'");
        return Ok(());
    }

    for manifest in manifests_to_pull {
        println!("pulling {} ({:.1} GB)...", manifest.name, manifest.size_gb);
        let downloads = pull_model(manifest, force).await?;
        println!(
            "  downloaded {} files for {}",
            downloads.len(),
            manifest.name
        );
    }

    // Persist config (ensures config file exists for future runs).
    let config = AiConfig::load().unwrap_or_default();
    config.save()?;

    println!("done.");
    Ok(())
}
