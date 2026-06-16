use crate::ModelManifest;
use nexo_core::ModelId;
use std::sync::LazyLock;

/// Flux2 Klein 9B model.
pub static FLUX2_KLEIN_9B: LazyLock<ModelManifest> =
    LazyLock::new(|| ModelManifest::new(ModelId::Flux2Klein9b, 4.0, vec![]));
