use serde::{Deserialize, Serialize};

/// Declares how a model adapter handles conversation roles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RoleStrategy {
    /// Preserve all roles as supplied by the caller.
    #[default]
    Default,

    /// Merge developer instructions into the system role before formatting.
    MergeDeveloperIntoSystem,
}
