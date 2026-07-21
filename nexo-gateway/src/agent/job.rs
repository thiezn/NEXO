use nexo_core::{ModelId, Node, OperationId, PeerId};
use strum::IntoStaticStr;

/// The category of agent job persisted in the database.
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
pub enum AgentJobKind {
    /// A job to run an inference request.
    RunInference,
}

/// A runnable job candidate selected from the SQLite scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunnableJobCandidate {
    /// Stable FIFO position assigned when the job is created.
    pub queue_position: i64,
    /// Stable operation identifier for the job.
    pub operation_id: OperationId,
    /// Owning user peer resolved through the operation row.
    pub user_peer_id: PeerId,
    /// Variant-specific reducer to invoke.
    pub kind: AgentJobKind,
}

/// A concrete node/model route considered by the inference scheduler.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InferenceRoutingCandidate {
    /// The target model is already loaded and inference can start immediately.
    Loaded {
        /// Node that already has the target model loaded.
        node: Node,
        /// Target model selected for inference.
        model_id: ModelId,
    },
    /// The node has no loaded models and can load the target directly.
    Load {
        /// Empty node selected for loading.
        node: Node,
        /// Target model to load.
        model_id: ModelId,
    },
    /// The node is occupied and must evict one model before loading the target.
    UnloadThenLoad {
        /// Occupied node selected for the operation.
        node: Node,
        /// Target model to load after eviction.
        model_id: ModelId,
        /// Loaded model selected for eviction.
        unloading_model_id: ModelId,
    },
}

impl InferenceRoutingCandidate {
    /// Return the stable preference key used to order routing candidates.
    pub(crate) fn sort_key(&self) -> (u8, String, String) {
        match self {
            Self::Loaded { node, model_id } => (0, String::from(*model_id), node.id().to_string()),
            Self::Load { node, model_id } => (1, String::from(*model_id), node.id().to_string()),
            Self::UnloadThenLoad { node, model_id, .. } => {
                (2, String::from(*model_id), node.id().to_string())
            }
        }
    }
}

/// Generic scheduler readiness for an agent job.
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
pub enum AgentJobSchedulerState {
    /// The variant reducer may advance the job on a queue tick.
    Runnable,
    /// One external operation is outstanding for the job.
    Waiting,
    /// The job completed successfully.
    Completed,
    /// The job completed with failure.
    Failed,
}
