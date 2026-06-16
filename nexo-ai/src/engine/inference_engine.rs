use super::{AnyTtsRuntime, MistralRsRuntime, MoldRuntime};
use crate::catalog::ModelManifest;
use crate::{Error, Result};
use nexo_core::{InferenceRequest, InferenceStream, ModelDefinition, ModelId, ModelRuntimeState};
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::{Mutex, watch};
use tracing::{info, warn};

/// The runtime implementation for a specific model.
///
/// Each model will have exactly one ModelRuntime instance, and will NOT be shared across multiple models.
/// This could cause some overhead but for now suspect this is negligible in comparison with the
/// actual model loaded in memory.
enum ModelRuntime {
    MistralRs(MistralRsRuntime),
    AnyTts(AnyTtsRuntime),
    Mold(MoldRuntime),
}

impl ModelRuntime {
    /// Loads the model into memory and starts its runtime, making it available for inference requests.
    async fn load_model(&mut self, model_id: &ModelId) -> Result {
        match self {
            ModelRuntime::MistralRs(runtime) => runtime.load_model(model_id).await,
            ModelRuntime::AnyTts(runtime) => runtime.load_model(model_id).await,
            ModelRuntime::Mold(runtime) => runtime.load_model(model_id).await,
        }
    }

    /// Unloads the model from memory and stops its runtime, freeing up resources.
    ///
    /// InferenceEngine is expected to deallocate the runtime after unloading.
    async fn unload_model(&mut self, model_id: &ModelId) -> Result {
        match self {
            ModelRuntime::MistralRs(runtime) => runtime.unload_model(model_id).await,
            ModelRuntime::AnyTts(runtime) => runtime.unload_model(model_id).await,
            ModelRuntime::Mold(runtime) => runtime.unload_model(model_id).await,
        }
    }

    /// Runs an inference request on the model and returns a stream of responses.
    async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match self {
            ModelRuntime::MistralRs(runtime) => runtime.infer(model_id, request).await,
            ModelRuntime::AnyTts(runtime) => runtime.infer(model_id, request).await,
            ModelRuntime::Mold(runtime) => runtime.infer(model_id, request).await,
        }
    }
}

/// A handle to a loaded model, containing its definition, a channel for
/// sending inference requests, and a watch channel for monitoring its runtime state.
///
/// All state transitions and inference requests for the model are routed through this handle
/// to ensure proper synchronization and state management.
struct ModelHandle {
    /// The static definition of the model, containing its capabilities and other metadata.
    definition: ModelDefinition,

    /// A channel for receiving ModelRuntimeState commands, such as load, unload, and run inference.
    state_rx: watch::Receiver<ModelRuntimeState>,

    /// A channel for sending commands to the model's runtime task, such as load, unload, and run inference.
    state_tx: watch::Sender<ModelRuntimeState>,

    /// A channel for sending commands to the model's runtime task, such as load, unload, and run inference.
    inner: Mutex<InnerModelHandle>,
}

/// The inner state of a ModelHandle, containing the command channel and state receiver for the model runtime task.
///
/// This separation allows the ModelHandle to provide a clean public API for interacting with the model, while
/// encapsulating the internal mutable state of the ModelHandle within a separate struct. The inner struct is wrapped in an Arc to allow for shared ownership and thread-safe access across multiple tasks.
struct InnerModelHandle {
    /// Current state of the model
    state: ModelRuntimeState,

    /// The runtime implementation for the model. We will instantiate multiple ModelRuntime instances
    /// instead of trying to leverage the multi-model support of MistralRs, as that would introduce
    /// unnecessary complexity.
    ///
    /// This choice might have to be revisited in the future if it turns out this causes a lot
    /// of overhead.
    runtime: ModelRuntime,
}

impl ModelHandle {
    /// Initializes a new ModelHandle for the given model definition, setting up the necessary channels and runtime task.
    pub fn new(definition: ModelDefinition) -> Result<Self> {
        let (state_tx, state_rx) = watch::channel(ModelRuntimeState::Unloaded);
        let runtime = match definition.id() {
            ModelId::Gemma426bA4bItUqffQ80
            | ModelId::EmbeddingGemma300m
            | ModelId::Gemma4E4bItUqffQ80 => ModelRuntime::MistralRs(MistralRsRuntime::new()),
            ModelId::Flux2Klein9b => ModelRuntime::Mold(MoldRuntime::new()),
            ModelId::Kokoro82m => ModelRuntime::AnyTts(AnyTtsRuntime::new()),
        };

        Ok(Self {
            definition,
            state_rx,
            state_tx,
            inner: Mutex::new(InnerModelHandle {
                state: ModelRuntimeState::Unloaded,
                runtime,
            }),
        })
    }

    /// Loads the model into memory and starts its runtime,
    /// making it available for inference requests.
    pub async fn load(&self) -> Result {
        let mut inner = self.inner.lock().await;

        match inner.state {
            ModelRuntimeState::Loaded => {
                warn!(model_id = %self.definition.id(), "Model is already loaded");
                return Ok(());
            }
            ModelRuntimeState::Unloaded | ModelRuntimeState::Failed => {
                inner.state = ModelRuntimeState::Loading;
                let _ = self.state_tx.send(inner.state);
            }
            _ => {
                return Err(Error::ModelNotUnloaded {
                    model_id: self.definition.id().clone(),
                    current_state: inner.state,
                });
            }
        }

        match inner.runtime.load_model(&self.definition.id()).await {
            Ok(()) => {
                inner.state = ModelRuntimeState::Loaded;
                let _ = self.state_tx.send(inner.state);
                Ok(())
            }
            Err(err) => {
                inner.state = ModelRuntimeState::Failed;
                let _ = self.state_tx.send(inner.state);
                Err(err)
            }
        }
    }

    /// Unloads the model from memory and stops its runtime,
    /// freeing up resources.
    pub async fn unload(&self) -> Result {
        let mut inner = self.inner.lock().await;

        match inner.state {
            ModelRuntimeState::Unloaded => {
                warn!(model_id = %self.definition.id(), "Model is already unloaded");
                return Ok(());
            }
            ModelRuntimeState::Loaded | ModelRuntimeState::Failed => {
                inner.state = ModelRuntimeState::Unloading;
                let _ = self.state_tx.send(inner.state);
            }
            _ => {
                return Err(Error::ModelNotUnloaded {
                    model_id: self.definition.id().clone(),
                    current_state: inner.state,
                });
            }
        }

        match inner.runtime.unload_model(&self.definition.id()).await {
            Ok(()) => {
                inner.state = ModelRuntimeState::Unloaded;
                let _ = self.state_tx.send(inner.state);
                Ok(())
            }
            Err(err) => {
                inner.state = ModelRuntimeState::Failed;
                let _ = self.state_tx.send(inner.state);
                Err(err)
            }
        }
    }

    /// Sends an inference request to the model and awaits the response.
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceStream> {
        // inner.lock().await will queue requests
        // let mut inner = self.inner.lock().await;

        // try_lock() will return an error if the model is busy.
        // For now I think this is what I want so the gateway gets a signal
        // it's trying to run inference on a model that is already busy.
        let mut inner = self.inner.try_lock().map_err(|_| Error::ModelBusy {
            model_id: self.definition.id().clone(),
        })?;

        if inner.state != ModelRuntimeState::Loaded {
            return Err(Error::ModelNotLoaded {
                model_id: self.definition.id().clone(),
                current_state: inner.state,
            });
        }

        inner.state = ModelRuntimeState::RunningInference;
        let _ = self.state_tx.send(inner.state);

        // NOTE: If result is an error, we are assuming the model runtime will still
        // be usable (loaded state) here. We might need to revisit this.
        let result = inner.runtime.infer(&self.definition.id(), request).await;
        inner.state = ModelRuntimeState::Loaded;
        let _ = self.state_tx.send(inner.state);

        result
    }
}

/// The InferenceEngine is responsible for managing the lifecycle of models,
/// including loading them into memory and starting inference on them.
///
/// ## Rules
///
/// - All provided models are expected to be pre-downloaded and available locally.
///
/// - Only one inference request can be active on a given model at a time.
///   Concurrent requests for the same model will be rejected with an appropriate error.
///
/// - Multiple inference requests can be active concurrently if they are targeting
///   different models. This allows for multi-modal scenarios where, for example,
///   both audio generation and text generation are running at the same time.
#[derive(Clone)]
pub struct InferenceEngine {
    /// The set of all models known to the engine, indexed by their unique identifier.
    models: BTreeMap<ModelId, Arc<ModelHandle>>,
}

impl InferenceEngine {
    /// Creates a new InferenceEngine.
    pub fn new(manifests: Vec<ModelManifest>) -> Result<Self> {
        let models = manifests
            .into_iter()
            .map(|m| {
                let definition = m.definition().clone();
                let model_id = m.model_id().clone();
                let handle = ModelHandle::new(definition)?;
                Ok((model_id, Arc::new(handle)))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self { models })
    }

    /// Load the given ModelId runtime in memory, making it available for inference.
    ///
    /// A single ModelId will always only support one runtime implementation.
    ///
    /// This allows us to avoid a lot of complexity around managing multiple
    /// runtime implementations for the same model.
    pub async fn load_model(&self, model_id: &ModelId) -> Result {
        if let Some(handle) = self.models.get(model_id) {
            handle.load().await
        } else {
            Err(Error::UnknownModel {
                model_id: model_id.clone(),
            })
        }
    }

    /// Unloads a model from memory, freeing up resources.
    pub async fn unload_model(&self, model_id: &ModelId) -> Result {
        if let Some(handle) = self.models.get(model_id) {
            handle.unload().await
        } else {
            Err(Error::UnknownModel {
                model_id: model_id.clone(),
            })
        }
    }

    /// Runs an incoming InferenceRequest by routing it to the appropriate model runtime based on the request's
    /// model selection criteria and the currently loaded models.
    ///
    /// Every inference request is mapped to a session. Within a session, multiple runs can be executed, and within each run,
    /// multiple rounds of inference can occur.
    ///
    /// Image/Video/Speech generation requests are expected to only run for a single round, while multi-modal requests may
    /// have multiple rounds (e.g. for multi-turn conversations).
    ///
    /// Management of the sessions, runs, and rounds is handled by the nexo-gateway agent loop. The InferenceEngine is only
    /// responsible for executing the inference based on the provided request.
    pub async fn run_inference(&self, request: InferenceRequest) -> Result<InferenceStream> {
        info!(request_id = %request.request_id, session_id = ?request.session_id, run_id = ?request.run_id, round_id = ?request.round_id, "Received inference request");

        // Route to the appropriate model runtime based on the payload's model selection criteria
        // and the currently loaded models.
        let model_id = request.model(self.model_definitions())?;

        if let Some(handle) = self.models.get(&model_id) {
            handle.infer(request).await
        } else {
            Err(Error::UnknownModel {
                model_id: model_id.clone(),
            })
        }
    }

    /// Returns a list of all model definitions known to the engine, based on the configured model manifests.
    fn model_definitions(&self) -> Vec<&ModelDefinition> {
        self.models
            .values()
            .map(|handle| &handle.definition)
            .collect()
    }
}
