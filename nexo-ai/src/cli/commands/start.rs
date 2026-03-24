use crate::cli::repl;
use crate::config::AiConfig;
use crate::coordinator::Coordinator;
use crate::shared::types::ModelCategory;
use anyhow::Result;

pub async fn run(categories: Option<Vec<String>>) -> Result<()> {
    let config = AiConfig::load().unwrap_or_default();
    let mut coordinator = Coordinator::new(config);

    // Load specified categories or startup defaults from config.
    if let Some(cats) = categories {
        let parsed: Vec<ModelCategory> = cats
            .iter()
            .filter_map(|s| {
                ModelCategory::all()
                    .iter()
                    .find(|c| c.as_str() == s)
                    .copied()
            })
            .collect();

        if !parsed.is_empty()
            && let Err(e) = coordinator.load_defaults(&parsed)
        {
            tracing::warn!("failed to load some models: {e}");
        }
    } else if let Err(e) = coordinator.load_startup_models() {
        tracing::warn!("failed to load startup models: {e}");
    }

    // Show what's loaded.
    let loaded = coordinator.loaded_models();
    if loaded.is_empty() {
        println!("no models loaded. Use /start models <category> to load models.");
    } else {
        println!("loaded models:");
        for (name, cats) in loaded {
            let cat_str: Vec<&str> = cats.iter().map(|c| c.as_str()).collect();
            println!("  {} [{}]", name, cat_str.join(", "));
        }
    }
    println!();

    // Enter REPL.
    repl::run_repl(&mut coordinator)
}
