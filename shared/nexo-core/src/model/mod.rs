//! Model metadata, capability declarations, and selection primitives.

/// Model capability declarations.
pub mod capability;
/// Model descriptor payloads.
pub mod descriptor;
/// Model modality declarations.
pub mod modality;
/// Thinking and reasoning configuration types.
pub mod reasoning;
/// Conversation role handling strategy declarations.
pub mod role_strategy;
/// Model runtime state types.
pub mod runtime_state;
/// Model selection criteria types.
pub mod selection;

pub use capability::ModelCapability;
pub use descriptor::ModelDescriptor;
pub use modality::{ModelModalities, SupportedModality};
pub use reasoning::{ReasoningEffort, ReasoningSettings, ThinkingMode};
pub use role_strategy::RoleStrategy;
pub use runtime_state::ModelRuntimeState;
pub use selection::{InferenceRuntime, ModelSelection};
