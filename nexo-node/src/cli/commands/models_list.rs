//! `models list` command implementation.

use nexo_ai::download::manifest::storage_path;
use nexo_ai::download::paths::default_models_dir;
use nexo_ai::registry::known_manifests;

/// Print the set of known models and whether each one is already downloaded.
///
/// # Errors
///
/// This command does not currently surface any fallible operations.
pub fn run() -> cli_helpers::Result {
    let manifests = known_manifests();

    if manifests.is_empty() {
        println!("No models registered.");
        return Ok(());
    }

    println!(
        "{:<30} {:<10} {:<12} DESCRIPTION",
        "NAME", "SIZE (GB)", "DOWNLOADED"
    );
    println!("{}", "-".repeat(80));

    let mdir = default_models_dir();

    for entry in manifests {
        let manifest = &entry.manifest;
        let downloaded = manifest
            .files
            .iter()
            .all(|f| mdir.join(storage_path(manifest, f)).exists());

        println!(
            "{:<30} {:<10.1} {:<12} {}",
            manifest.name,
            manifest.size_gb,
            if downloaded { "yes" } else { "no" },
            manifest.description,
        );
    }

    Ok(())
}
