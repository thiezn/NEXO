use crate::config::AiConfig;
use crate::coordinator::Coordinator;
use anyhow::Result;

pub fn run() -> Result<()> {
    let config = AiConfig::load().unwrap_or_default();
    let coordinator = Coordinator::new(config);
    let models = coordinator.list_models();

    if models.is_empty() {
        println!("no models registered. Model implementations will be added in future updates.");
        return Ok(());
    }

    // Print header.
    println!(
        "{:<25} {:<15} {:<15} {:<8} {:<12} DESCRIPTION",
        "NAME", "FAMILY", "CATEGORIES", "SIZE", "DOWNLOADED"
    );
    println!("{}", "-".repeat(95));

    for model in models {
        let cats: Vec<&str> = model.categories.iter().map(|c| c.as_str()).collect();
        let downloaded = if model.is_downloaded { "yes" } else { "no" };
        println!(
            "{:<25} {:<15} {:<15} {:<8} {:<12} {}",
            model.name,
            model.family,
            cats.join(","),
            format!("{:.1}G", model.size_gb),
            downloaded,
            model.description
        );
    }

    Ok(())
}
