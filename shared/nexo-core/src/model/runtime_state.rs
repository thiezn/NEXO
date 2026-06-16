use serde::{Deserialize, Serialize};

/// The current runtime availability state of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelRuntimeState {
    /// The model is currently Unloading
    Unloading,

    /// The model is not currently loaded into the runtime.
    Unloaded,

    /// The model is being loaded and is not ready yet.
    Loading,

    /// The model is loaded and ready to serve requests.
    Loaded,

    /// The model is currently running an inference request
    RunningInference,

    /// The model failed to become available.
    Failed,
}
