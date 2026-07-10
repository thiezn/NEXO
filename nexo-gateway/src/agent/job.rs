use nexo_core::{InferenceIntent, OperationId, PeerId};
use strum::{EnumDiscriminants, IntoStaticStr};

/// A single job that the NexoAgent can perform.
///
/// These jobs are queued up in the NexoAgent queue. The agent is responsible
/// for handling parallelism and sequencing.
#[derive(Debug, IntoStaticStr, PartialEq, EnumDiscriminants)]
#[strum_discriminants(name(AgentJobKind))]
#[strum_discriminants(vis(pub))]
#[strum_discriminants(doc = "The category of agent job persisted in the database.")]
#[strum_discriminants(derive(
    Hash,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr
))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
pub(crate) enum AgentJob {
    /// A job to run an inference request.
    RunInference {
        /// Stable operation identifier for the job.
        operation_id: OperationId,
        /// Owning user peer for the job.
        user_peer_id: PeerId,
        /// Original inference intent payload.
        intent: InferenceIntent,
    },
}

impl From<(PeerId, InferenceIntent)> for AgentJob {
    fn from((user_peer_id, intent): (PeerId, InferenceIntent)) -> Self {
        Self::RunInference {
            operation_id: intent.operation_id,
            user_peer_id,
            intent,
        }
    }
}

/// The persisted queue lifecycle of an agent job.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum AgentJobQueueStatus {
    /// The job is waiting to be claimed.
    Queued,
    /// The job has been claimed for processing.
    Claimed,
    /// The job completed successfully.
    Completed,
    /// The job completed with failure.
    Failed,
}
