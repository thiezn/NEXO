//! All the ModelDefinitions separated by ModelFamily

use crate::ModelManifest;
use std::sync::LazyLock;

/// All Embedding Gamma models
pub mod embeddinggemma;

/// All Flux2 models
pub mod flux2;

/// All Gemma 4 models
pub mod gemma4;

/// All Kokoro models
pub mod kokoro;

/// A static list of all supported ModelManifests
pub static ALL_MANIFESTS: [&LazyLock<ModelManifest>; 7] = [
    &embeddinggemma::EMBEDDING_GEMMA_300M,
    &flux2::FLUX2_KLEIN_9B,
    &gemma4::GEMMA_4_26B_A4B_IT_UQFF_AFQ6,
    &gemma4::GEMMA_4_26B_A4B_IT_UQFF_AFQ8,
    &gemma4::GEMMA_4_E4B_IT_UQFF_AFQ6,
    &gemma4::GEMMA_4_E4B_IT_UQFF_AFQ8,
    &kokoro::KOKORO_82M,
];
