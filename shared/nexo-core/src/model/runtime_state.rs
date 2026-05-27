use serde::{Deserialize, Serialize};

/// The current runtime availability state of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ModelRuntimeState {
    /// The model is not currently loaded into the runtime.
    Unloaded,

    /// The model is being loaded and is not ready yet.
    Loading,

    /// The model is loaded and ready to serve requests.
    Loaded,

    /// The model is reloading after an unload or configuration change.
    Reloading,

    /// The model failed to become available.
    Failed,
}
