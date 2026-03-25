use crate::cli::repl;
use crate::config::AiConfig;
use crate::coordinator::Coordinator;
use crate::shared::types::ModelCategory;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
            && let Err(e) = coordinator.load_active_models(&parsed)
        {
            tracing::warn!("failed to load some models: {e}");
        }
    } else if let Err(e) = coordinator.load_startup_categories() {
        tracing::warn!("failed to load startup models: {e}");
    }

    // Show what's loaded.
    let loaded = coordinator.loaded_models();
    if loaded.is_empty() {
        println!("no models loaded. Use /load categories <category> to load models.");
    } else {
        println!("loaded models:");
        for (name, cats) in loaded {
            let cat_str: Vec<&str> = cats.iter().map(|c| c.as_str()).collect();
            println!("  {} [{}]", name, cat_str.join(", "));
        }
    }
    println!();

    // Set up Ctrl+C handler.
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_hook = shutdown.clone();
    ctrlc::set_handler(move || {
        if shutdown_hook.swap(true, Ordering::SeqCst) {
            // Second Ctrl+C — force exit.
            std::process::exit(1);
        }
        eprintln!("\nreceived ctrl+c, press again to force quit...");
    })
    .ok();

    // Enter REPL.
    repl::run_repl(&mut coordinator, shutdown)
}
