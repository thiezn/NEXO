//! Durable inference-run lifecycle types used by the scheduler and persistence boundary.

use nexo_core::{ModelId, OperationId, PeerId};
use strum::EnumDiscriminants;

/// Application-facing state reconstructed from one persisted inference run.
#[derive(Debug, Clone, PartialEq, Eq, EnumDiscriminants)]
#[strum_discriminants(name(InferenceRunStateKind))]
#[strum_discriminants(vis(pub))]
#[strum_discriminants(doc = "The persisted lifecycle discriminant for an inference run.")]
#[strum_discriminants(derive(
    Hash,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr
))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
pub enum InferenceRunState {
    /// The request is accepted but context preparation has not started.
    Queued,
    /// Context preparation started and no node or model has been selected.
    PreparingContext,
    /// A leased node is evicting one loaded model before loading the target model.
    UnloadingModel {
        /// Node leased by the inference job.
        node_peer_id: PeerId,
        /// Target model that the job will load after eviction.
        model_id: ModelId,
        /// Loaded model that must be evicted from the node.
        unloading_model_id: ModelId,
    },
    /// A leased node is loading the selected target model, or the load completed and the job is runnable.
    LoadingModel {
        /// Node leased by the inference job.
        node_peer_id: PeerId,
        /// Target model selected for the inference run.
        model_id: ModelId,
    },
    /// The selected node has been asked to execute the inference request.
    InProgress {
        /// Node leased by the inference job.
        node_peer_id: PeerId,
        /// Model executing the inference request.
        model_id: ModelId,
    },
    /// The inference run completed successfully.
    Completed {
        /// Node that executed the inference request.
        node_peer_id: PeerId,
        /// Model that executed the inference request.
        model_id: ModelId,
    },
    /// The inference run failed before or after node selection.
    Failed {
        /// Human-readable failure detail.
        error_message: String,
        /// Selected node, if routing had completed.
        node_peer_id: Option<PeerId>,
        /// Selected target model, if routing had completed.
        model_id: Option<ModelId>,
    },
}

impl InferenceRunState {
    /// Return the stable persisted discriminant for this run state.
    pub fn kind(&self) -> InferenceRunStateKind {
        InferenceRunStateKind::from(self)
    }
}

/// Persisted lifecycle timestamps for one inference run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunTimeline {
    /// When the run row was created.
    pub created_at: String,
    /// When context preparation began.
    pub preparing_started_at: Option<String>,
    /// When a node and target model were selected.
    pub node_selected_at: Option<String>,
    /// When target-model loading began.
    pub model_loading_started_at: Option<String>,
    /// When inference dispatch began.
    pub in_progress_at: Option<String>,
    /// When the run completed successfully.
    pub completed_at: Option<String>,
    /// When the run failed.
    pub failed_at: Option<String>,
    /// When the current persisted state was last changed.
    pub last_state_changed_at: String,
}

/// Application-facing snapshot reconstructed from one persisted inference-run row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunSnapshot {
    /// Stable operation identifier for the run.
    pub operation_id: OperationId,
    /// Current durable workflow state.
    pub state: InferenceRunState,
    /// Durable lifecycle timestamps for the run.
    pub timeline: InferenceRunTimeline,
}
