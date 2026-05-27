use serde::{Deserialize, Serialize};

/// Controls how the model may use available tools during generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Disable tool use for the request.
    #[default]
    Disabled,

    /// Let the model choose whether to call a tool.
    Automatic,

    /// Force a specific tool to be selected.
    Specific {
        /// The name of the tool that must be called.
        name: String,
    },
}
