//! Typestate-based inference run lifecycle primitives.

use nexo_core::{ModelId, OperationId, PeerId};
use strum::EnumDiscriminants;

/// Shared runtime carrier for a single inference run at a specific lifecycle stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRun<S> {
    operation_id: OperationId,
    user_peer_id: PeerId,
    state: S,
}

impl<S> InferenceRun<S> {
    /// Return the stable operation identifier for the run.
    pub fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    /// Return the owning user peer for the run.
    pub fn user_peer_id(&self) -> PeerId {
        self.user_peer_id
    }

    /// Borrow the stage-specific state payload.
    pub fn state(&self) -> &S {
        &self.state
    }
}

/// Marker state for a run that has been accepted and queued, but not started yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Queued;

/// Marker state for the context-preparation phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreparingContext;

/// Marker state for a run that must first unload a conflicting model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnloadingModel {
    node_peer_id: PeerId,
    model_id: ModelId,
}

impl UnloadingModel {
    /// Return the selected node for the run.
    pub fn node_peer_id(&self) -> PeerId {
        self.node_peer_id
    }

    /// Return the selected model for the run.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }
}

/// Marker state for a run that is loading its model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadingModel {
    node_peer_id: PeerId,
    model_id: ModelId,
}

impl LoadingModel {
    /// Return the selected node for the run.
    pub fn node_peer_id(&self) -> PeerId {
        self.node_peer_id
    }

    /// Return the selected model for the run.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }
}

/// Marker state for a run actively executing on a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InProgress {
    node_peer_id: PeerId,
    model_id: ModelId,
}

impl InProgress {
    /// Return the selected node for the run.
    pub fn node_peer_id(&self) -> PeerId {
        self.node_peer_id
    }

    /// Return the selected model for the run.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }
}

/// Marker state for a run that completed successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completed {
    node_peer_id: PeerId,
    model_id: ModelId,
}

impl Completed {
    /// Return the selected node for the run.
    pub fn node_peer_id(&self) -> PeerId {
        self.node_peer_id
    }

    /// Return the selected model for the run.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }
}

/// Marker state for a run that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failed {
    error_message: String,
    node_peer_id: Option<PeerId>,
    model_id: Option<ModelId>,
}

impl Failed {
    /// Return the human-readable failure detail.
    pub fn error_message(&self) -> &str {
        &self.error_message
    }

    /// Return the selected node for the run, if one had already been chosen.
    pub fn node_peer_id(&self) -> Option<PeerId> {
        self.node_peer_id
    }

    /// Return the selected model for the run, if one had already been chosen.
    pub fn model_id(&self) -> Option<ModelId> {
        self.model_id
    }
}

/// Application-facing state snapshot used at the DB boundary.
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
    /// The request has been accepted and queued, but preparation has not started yet.
    Queued,
    /// Context preparation has started and no node/model has been selected yet.
    PreparingContext,
    /// A node and model are selected and the node is unloading its current model.
    UnloadingModel {
        /// Selected node for the run.
        node_peer_id: PeerId,
        /// Selected model for the run.
        model_id: ModelId,
    },
    /// A node and model are selected and the model is being loaded.
    LoadingModel {
        /// Selected node for the run.
        node_peer_id: PeerId,
        /// Selected model for the run.
        model_id: ModelId,
    },
    /// A node and model are selected and the run is in progress.
    InProgress {
        /// Selected node for the run.
        node_peer_id: PeerId,
        /// Selected model for the run.
        model_id: ModelId,
    },
    /// The run completed successfully.
    Completed {
        /// Selected node for the completed run.
        node_peer_id: PeerId,
        /// Selected model for the completed run.
        model_id: ModelId,
    },
    /// The run failed, optionally after node/model selection.
    Failed {
        /// Human-readable failure detail.
        error_message: String,
        /// Selected node for the failed run, if one had been chosen.
        node_peer_id: Option<PeerId>,
        /// Selected model for the failed run, if one had been chosen.
        model_id: Option<ModelId>,
    },
}

impl InferenceRunState {
    /// Return the persisted discriminant for this state.
    pub fn kind(&self) -> InferenceRunStateKind {
        InferenceRunStateKind::from(self)
    }
}

/// Persisted lifecycle timestamps for a single run row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunTimeline {
    /// When the run row was first created.
    pub created_at: String,
    /// When context preparation began.
    pub preparing_started_at: Option<String>,
    /// When a node/model selection was first persisted.
    pub node_selected_at: Option<String>,
    /// When model loading started.
    pub model_loading_started_at: Option<String>,
    /// When inference execution started.
    pub in_progress_at: Option<String>,
    /// When the run completed successfully.
    pub completed_at: Option<String>,
    /// When the run failed.
    pub failed_at: Option<String>,
    /// When the current persisted state was last updated.
    pub last_state_changed_at: String,
}

/// Application-facing view of the currently persisted inference run row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunSnapshot {
    /// Stable operation identifier for the run.
    pub operation_id: OperationId,
    /// Current persisted run state.
    pub state: InferenceRunState,
    /// Lifecycle timestamps persisted for the run.
    pub timeline: InferenceRunTimeline,
}

impl InferenceRun<Queued> {
    /// Start tracking a newly queued inference run.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The stable operation identifier for the run.
    /// * `user_peer_id` - The owning user peer for the run.
    pub fn new(operation_id: OperationId, user_peer_id: PeerId) -> Self {
        Self {
            operation_id,
            user_peer_id,
            state: Queued,
        }
    }

    /// Transition from queued into context preparation.
    pub fn into_preparing_context(self) -> InferenceRun<PreparingContext> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: PreparingContext,
        }
    }

    /// Transition into failure before any preparation starts.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The failure message to persist for the run.
    pub fn into_failed(self, error_message: impl Into<String>) -> InferenceRun<Failed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Failed {
                error_message: error_message.into(),
                node_peer_id: None,
                model_id: None,
            },
        }
    }
}

impl InferenceRun<PreparingContext> {
    /// Transition directly into model unloading after selecting a node and model.
    ///
    /// # Arguments
    ///
    /// * `node_peer_id` - The selected node for the run.
    /// * `model_id` - The selected model for the run.
    pub fn into_unloading_model(
        self,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> InferenceRun<UnloadingModel> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: UnloadingModel {
                node_peer_id,
                model_id,
            },
        }
    }

    /// Transition directly into model loading after selecting a node and model.
    ///
    /// # Arguments
    ///
    /// * `node_peer_id` - The selected node for the run.
    /// * `model_id` - The selected model for the run.
    pub fn into_loading_model(
        self,
        node_peer_id: PeerId,
        model_id: ModelId,
    ) -> InferenceRun<LoadingModel> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: LoadingModel {
                node_peer_id,
                model_id,
            },
        }
    }

    /// Transition into failure before any node/model has been selected.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The failure message to persist for the run.
    pub fn into_failed(self, error_message: impl Into<String>) -> InferenceRun<Failed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Failed {
                error_message: error_message.into(),
                node_peer_id: None,
                model_id: None,
            },
        }
    }
}

impl InferenceRun<UnloadingModel> {
    /// Continue from unloading into loading.
    pub fn into_loading_model(self) -> InferenceRun<LoadingModel> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: LoadingModel {
                node_peer_id: self.state.node_peer_id,
                model_id: self.state.model_id,
            },
        }
    }

    /// Transition into failure after selection.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The failure message to persist for the run.
    pub fn into_failed(self, error_message: impl Into<String>) -> InferenceRun<Failed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Failed {
                error_message: error_message.into(),
                node_peer_id: Some(self.state.node_peer_id),
                model_id: Some(self.state.model_id),
            },
        }
    }
}

impl InferenceRun<LoadingModel> {
    /// Continue from loading into active execution.
    pub fn into_in_progress(self) -> InferenceRun<InProgress> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: InProgress {
                node_peer_id: self.state.node_peer_id,
                model_id: self.state.model_id,
            },
        }
    }

    /// Transition into failure after selection.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The failure message to persist for the run.
    pub fn into_failed(self, error_message: impl Into<String>) -> InferenceRun<Failed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Failed {
                error_message: error_message.into(),
                node_peer_id: Some(self.state.node_peer_id),
                model_id: Some(self.state.model_id),
            },
        }
    }
}

impl InferenceRun<InProgress> {
    /// Mark the run completed successfully.
    pub fn into_completed(self) -> InferenceRun<Completed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Completed {
                node_peer_id: self.state.node_peer_id,
                model_id: self.state.model_id,
            },
        }
    }

    /// Transition into failure during execution.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The failure message to persist for the run.
    pub fn into_failed(self, error_message: impl Into<String>) -> InferenceRun<Failed> {
        InferenceRun {
            operation_id: self.operation_id,
            user_peer_id: self.user_peer_id,
            state: Failed {
                error_message: error_message.into(),
                node_peer_id: Some(self.state.node_peer_id),
                model_id: Some(self.state.model_id),
            },
        }
    }
}

impl From<&InferenceRun<Queued>> for InferenceRunState {
    fn from(_value: &InferenceRun<Queued>) -> Self {
        Self::Queued
    }
}

impl From<&InferenceRun<PreparingContext>> for InferenceRunState {
    fn from(_value: &InferenceRun<PreparingContext>) -> Self {
        Self::PreparingContext
    }
}

impl From<&InferenceRun<UnloadingModel>> for InferenceRunState {
    fn from(value: &InferenceRun<UnloadingModel>) -> Self {
        Self::UnloadingModel {
            node_peer_id: value.state.node_peer_id,
            model_id: value.state.model_id,
        }
    }
}

impl From<&InferenceRun<LoadingModel>> for InferenceRunState {
    fn from(value: &InferenceRun<LoadingModel>) -> Self {
        Self::LoadingModel {
            node_peer_id: value.state.node_peer_id,
            model_id: value.state.model_id,
        }
    }
}

impl From<&InferenceRun<InProgress>> for InferenceRunState {
    fn from(value: &InferenceRun<InProgress>) -> Self {
        Self::InProgress {
            node_peer_id: value.state.node_peer_id,
            model_id: value.state.model_id,
        }
    }
}

impl From<&InferenceRun<Completed>> for InferenceRunState {
    fn from(value: &InferenceRun<Completed>) -> Self {
        Self::Completed {
            node_peer_id: value.state.node_peer_id,
            model_id: value.state.model_id,
        }
    }
}

impl From<&InferenceRun<Failed>> for InferenceRunState {
    fn from(value: &InferenceRun<Failed>) -> Self {
        Self::Failed {
            error_message: value.state.error_message.clone(),
            node_peer_id: value.state.node_peer_id,
            model_id: value.state.model_id,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, User, UserProperties};

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    #[test]
    fn typestate_transitions_preserve_identity_and_selection() {
        let user = test_user();
        let operation_id = OperationId::new();
        let node_peer_id = user.id();
        let queued = InferenceRun::new(operation_id, user.id());

        let preparing = queued.into_preparing_context();
        let unloading = preparing.into_unloading_model(node_peer_id, ModelId::Kokoro82m);
        let loading = unloading.into_loading_model();
        let in_progress = loading.into_in_progress();
        let completed = in_progress.into_completed();

        assert_eq!(completed.operation_id(), operation_id);
        assert_eq!(completed.user_peer_id(), user.id());
        assert_eq!(completed.state().node_peer_id(), node_peer_id);
        assert_eq!(completed.state().model_id(), ModelId::Kokoro82m);
        assert_eq!(
            InferenceRunState::from(&completed).kind(),
            InferenceRunStateKind::Completed
        );
    }

    #[test]
    fn failed_state_preserves_optional_selection() {
        let user = test_user();
        let operation_id = OperationId::new();
        let queued = InferenceRun::new(operation_id, user.id());
        let failed = queued.into_failed("routing failed");

        let state = InferenceRunState::from(&failed);
        assert_eq!(state.kind(), InferenceRunStateKind::Failed);

        let InferenceRunState::Failed {
            error_message,
            node_peer_id,
            model_id,
        } = state
        else {
            panic!("expected failed state")
        };

        assert_eq!(error_message, "routing failed");
        assert_eq!(node_peer_id, None);
        assert_eq!(model_id, None);
    }
}
