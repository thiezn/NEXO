//! Agent runtime coordination and inference lifecycle module wiring.

pub mod inference_run_state;
mod job;
mod messages;
mod runtime;

pub use inference_run_state::{
    Completed, Failed, InProgress, InferenceRun, InferenceRunSnapshot, InferenceRunState,
    InferenceRunStateKind, InferenceRunTimeline, LoadingModel, PreparingContext, Queued,
    UnloadingModel,
};
pub use job::{AgentJobKind, AgentJobQueueStatus};
pub use messages::{NexoAgentInput, NexoAgentOutput};
pub use runtime::NexoAgent;
