//! Shared test helpers for nexo-ai integration tests.
#![allow(dead_code)]

use std::sync::Once;

use nexo_ai::download::manifest::storage_path;
use nexo_ai::download::paths::{default_models_dir, model_storage_dir};
use nexo_ai::registry::manifest::find_manifest;

static INIT_TRACING: Once = Once::new();

/// Initialize tracing for integration tests. Debug level for nexo_ai, info for everything else.
/// Call at the start of each test; `Once` ensures it only runs once per test binary.
pub fn init_tracing() {
    INIT_TRACING.call_once(|| {
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,nexo_ai=debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

/// Resolve model directory, panicking with download instructions if missing.
/// Also verifies all expected files from the manifest are present.
pub fn resolve_model(model_name: &str) -> (std::path::PathBuf, u64) {
    init_tracing();

    let manifest = find_manifest(model_name)
        .unwrap_or_else(|| panic!("unknown model '{model_name}' in manifest registry"));

    let dir = model_storage_dir(model_name);
    if !dir.exists() {
        panic!(
            "\n\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  MODEL NOT DOWNLOADED: {:<37} ║\n\
             ╠══════════════════════════════════════════════════════════════╣\n\
             ║  Expected directory:                                        ║\n\
             ║    {}\n\
             ║                                                              ║\n\
             ║  Download with:                                              ║\n\
             ║    cargo run -p nexo-ai --features cli -- pull {}\n\
             ╚══════════════════════════════════════════════════════════════╝\n",
            model_name,
            dir.display(),
            model_name,
        );
    }

    // Verify all expected files from the manifest are present
    let models_dir = default_models_dir();
    let missing: Vec<_> = manifest
        .manifest
        .files
        .iter()
        .filter(|f| {
            let path = models_dir.join(storage_path(&manifest.manifest, f));
            !path.exists()
        })
        .map(|f| f.hf_filename.as_str())
        .collect();

    if !missing.is_empty() {
        panic!(
            "\n\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  MODEL INCOMPLETE: {:<40} ║\n\
             ╠══════════════════════════════════════════════════════════════╣\n\
             ║  Missing files:                                              ║\n\
             {}\
             ║                                                              ║\n\
             ║  Re-download with:                                           ║\n\
             ║    cargo run -p nexo-ai --features cli -- pull {} --force\n\
             ╚══════════════════════════════════════════════════════════════╝\n",
            model_name,
            missing
                .iter()
                .map(|f| format!("║    - {f}\n"))
                .collect::<String>(),
            model_name,
        );
    }

    let memory_bytes = (manifest.manifest.size_gb * 1_000_000_000.0) as u64;
    (dir, memory_bytes)
}

/// Create a small test PNG image (solid red 64x64).
pub fn create_test_png() -> Vec<u8> {
    let mut buf = Vec::new();
    let img = image::RgbImage::from_fn(64, 64, |_, _| image::Rgb([255u8, 0, 0]));
    let dyn_img = image::DynamicImage::ImageRgb8(img);
    let mut cursor = std::io::Cursor::new(&mut buf);
    dyn_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("failed to write test image");
    buf
}
