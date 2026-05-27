use serde::{Deserialize, Serialize};

/// Controls whether explicit model thinking is enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// Do not request hidden or structured thinking output.
    #[default]
    Disabled,

    /// Enable model thinking for supported runtimes.
    Enabled,
}

/// The requested reasoning effort for models that expose effort tuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Request minimal reasoning for lower latency.
    Low,

    /// Request balanced reasoning depth.
    #[default]
    Medium,

    /// Request deeper reasoning for more thorough analysis.
    High,
}

/// The reasoning controls attached to a generation request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ReasoningSettings {
    /// Whether thinking is enabled for the request.
    pub thinking: ThinkingMode,

    /// The requested reasoning effort, if the target model supports it.
    pub effort: Option<ReasoningEffort>,
}
