use crate::download::{default_models_dir, known_manifests, storage_path};

pub fn run() -> utl_helpers::Result {
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

    for manifest in manifests {
        let downloaded = manifest.files.iter().all(|f| {
            let path = mdir.join(storage_path(manifest, f));
            path.exists()
        });

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
