//! All the ModelDefinitions separated by ModelFamily

use mold_ai_core::manifest::ModelManifest;

pub mod dia;
pub mod flux2;
pub mod gemma4;

/// A static list of all supported ModelManifests
pub const ALL_MANIFESTS: &[ModelManifest] = &[];
