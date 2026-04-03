pub mod manifest;
pub mod models;

pub use manifest::{
    AiComponent, AiModelManifest, find_manifest, known_manifests, manifests_for_category,
};
pub use models::{ModelEntry, list_models};

/// Scan the models directory to find which registered models are fully downloaded
/// and have valid SHA-256 checksums (where checksums are declared in the manifest).
#[cfg(feature = "download")]
pub fn detect_available_models() -> Vec<String> {
    use crate::download::manifest::storage_path;
    use crate::download::paths::default_models_dir;
    use rayon::prelude::*;

    let mdir = default_models_dir();
    known_manifests()
        .par_iter()
        .filter_map(|m| {
            let files_valid = m.manifest.files.par_iter().all(|f| {
                let path = mdir.join(storage_path(&m.manifest, f));
                if !path.exists() {
                    return false;
                }
                if let Some(expected_hash) = f.sha256 {
                    match crate::download::verify_sha256(&path, expected_hash) {
                        Ok(true) => true,
                        Ok(false) => {
                            tracing::warn!(
                                "SHA-256 mismatch for {} in model '{}' — file may be corrupt",
                                f.hf_filename,
                                m.manifest.name,
                            );
                            false
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to verify SHA-256 for {} in model '{}': {e}",
                                f.hf_filename,
                                m.manifest.name,
                            );
                            false
                        }
                    }
                } else {
                    true
                }
            });
            if files_valid {
                Some(m.manifest.name.clone())
            } else {
                None
            }
        })
        .collect()
}
