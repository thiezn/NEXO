//! Model metadata, capability declarations, and selection primitives.

/// Model capability declarations.
pub mod capability;
/// Model descriptor payloads.
pub mod definition;
/// Thinking and reasoning configuration types.
pub mod reasoning;
/// Conversation role handling strategy declarations.
pub mod role_strategy;
/// Model runtime state types.
pub mod runtime_state;

/// Model family types.
pub mod family;

//

pub use capability::ModelCapability;
pub use definition::ModelDefinition;
pub use family::ModelFamily;
pub use reasoning::{ReasoningEffort, ReasoningSettings, ThinkingMode};
pub use role_strategy::RoleStrategy;
pub use runtime_state::ModelRuntimeState;
