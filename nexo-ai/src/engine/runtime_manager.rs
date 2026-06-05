use std::collections::{BTreeMap, BTreeSet};

use nexo_core::inference::request::GenerateRequest;
use nexo_core::{
    InferenceRequest, InferenceRuntime, InferenceStream, ModelDescriptor, ModelId, ModelSelection,
};

use crate::{
    Error, InferenceEngineConfig, RegisteredModelConfig, Result, RuntimeConfig, StaticModelRegistry,
};

use super::any_tts::{AnyTtsRuntime, internal_runtime_kind};
use super::mistralrs::MistralRuntime;
use super::mold::MoldRuntime;

#[derive(Debug)]
pub(crate) struct RuntimeManager {
    runtime_config: RuntimeConfig,
    slots: BTreeMap<ModelId, ModelSlot>,
}

#[derive(Debug, Clone)]
struct ModelSlot {
    descriptor: ModelDescriptor,
    config: RegisteredModelConfig,
    state: ModelSlotState,
}

#[derive(Debug, Clone)]
enum ModelSlotState {
    Unloaded,
    Loaded(ActiveModelRuntime),
}

#[derive(Debug, Clone)]
struct ActiveModelRuntime {
    runtime_kind: InferenceRuntime,
    runtime: BackendRuntime,
    active_session_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum BackendRuntime {
    AnyTts(AnyTtsRuntime),
    MistralRs(MistralRuntime),
    Mold(MoldRuntime),
}

impl BackendRuntime {
    pub(crate) async fn submit(
        &self,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match self {
            Self::AnyTts(runtime) => runtime.submit(descriptor, request).await,
            Self::MistralRs(runtime) => runtime.submit(descriptor, request).await,
            Self::Mold(runtime) => runtime.submit(descriptor, request).await,
        }
    }
}

impl RuntimeManager {
    pub(crate) fn new(config: &InferenceEngineConfig) -> Self {
        Self {
            runtime_config: config.runtime.clone(),
            slots: config
                .models
                .iter()
                .cloned()
                .map(|model| {
                    (
                        model.descriptor.id.clone(),
                        ModelSlot {
                            descriptor: model.descriptor.clone(),
                            config: model,
                            state: ModelSlotState::Unloaded,
                        },
                    )
                })
                .collect(),
        }
    }

    pub(crate) fn loaded_model_ids(&self) -> BTreeSet<ModelId> {
        self.slots
            .iter()
            .filter_map(|(model_id, slot)| {
                matches!(slot.state, ModelSlotState::Loaded(_)).then(|| model_id.clone())
            })
            .collect()
    }

    pub(crate) async fn load_model(
        &mut self,
        model_id: &ModelId,
        runtime_kind: InferenceRuntime,
    ) -> Result<()> {
        let runtime_config = self.runtime_config.clone();
        let (config, resolved_runtime_kind) = match self.slots.get(model_id) {
            Some(slot) => (
                slot.config.clone(),
                Self::resolve_load_runtime_kind(&slot.config, runtime_kind)?,
            ),
            None => {
                return Err(Error::UnknownModel {
                    model_id: model_id.clone(),
                });
            }
        };

        if matches!(
            self.slots.get(model_id),
            Some(ModelSlot {
                state: ModelSlotState::Loaded(ActiveModelRuntime {
                    runtime_kind: active_runtime_kind,
                    ..
                }),
                ..
            }) if *active_runtime_kind == resolved_runtime_kind
        ) {
            return Ok(());
        }

        let runtime = Self::build_runtime(&runtime_config, &config, resolved_runtime_kind).await?;
        let slot = self.slot_mut(model_id)?;
        slot.state = ModelSlotState::Loaded(ActiveModelRuntime {
            runtime_kind: resolved_runtime_kind,
            runtime,
            active_session_id: None,
        });
        Ok(())
    }

    pub(crate) fn unload_model(&mut self, model_id: &ModelId) -> Result<bool> {
        let slot = self.slot_mut(model_id)?;
        let was_loaded = matches!(slot.state, ModelSlotState::Loaded(_));
        slot.state = ModelSlotState::Unloaded;
        Ok(was_loaded)
    }

    pub(crate) async fn prepare_request(
        &mut self,
        request: &InferenceRequest,
    ) -> Result<(ModelDescriptor, BackendRuntime)> {
        let descriptor = self.resolve_loaded_model(request)?;
        let model_id = descriptor.id.clone();
        let requested_session_id = request_session_id(request).map(str::to_owned);
        let requested_runtime_kind = requested_runtime_kind(request);
        let runtime_config = self.runtime_config.clone();

        let (config, state) = {
            let slot = self.slot_mut(&model_id)?;
            let state = std::mem::replace(&mut slot.state, ModelSlotState::Unloaded);
            (slot.config.clone(), state)
        };

        let ModelSlotState::Loaded(mut loaded) = state else {
            return Err(Error::ModelNotLoaded { model_id });
        };

        let next_runtime_kind = requested_runtime_kind.unwrap_or(loaded.runtime_kind);
        let runtime = if should_reload_for_session(request, loaded.active_session_id.as_deref())
            || should_switch_runtime(requested_runtime_kind, loaded.runtime_kind)
        {
            tracing::info!(
                model_id = %model_id,
                previous_runtime_kind = ?loaded.runtime_kind,
                requested_runtime_kind = ?requested_runtime_kind,
                previous_session_id = ?loaded.active_session_id,
                requested_session_id = ?requested_session_id,
                "Reloading model runtime for session change"
            );
            drop(loaded);
            Self::build_runtime(&runtime_config, &config, next_runtime_kind).await?
        } else {
            if matches!(request, InferenceRequest::Generate(_)) {
                loaded.active_session_id = requested_session_id.clone();
            }
            let runtime = loaded.runtime.clone();
            let slot = self.slot_mut(&model_id)?;
            slot.state = ModelSlotState::Loaded(loaded);
            return Ok((descriptor, runtime));
        };

        let slot = self.slot_mut(&model_id)?;
        slot.state = ModelSlotState::Loaded(ActiveModelRuntime {
            runtime_kind: next_runtime_kind,
            runtime: runtime.clone(),
            active_session_id: requested_session_id,
        });
        Ok((descriptor, runtime))
    }

    fn resolve_loaded_model(&self, request: &InferenceRequest) -> Result<ModelDescriptor> {
        let selection = request_model_selection(request);

        if let Some(model_id) = &selection.specific_model {
            let slot = self
                .slots
                .get(model_id)
                .ok_or_else(|| Error::UnknownModel {
                    model_id: model_id.clone(),
                })?;
            if !supports_selection(&slot.descriptor, selection) {
                return Err(Error::UnresolvedModelSelection {
                    message: format!(
                        "loaded model `{model_id}` does not satisfy the requested capabilities"
                    ),
                });
            }
            if !matches_runtime_preference(&slot.state, selection.runtime_preference) {
                return Err(Error::UnresolvedModelSelection {
                    message: format!(
                        "loaded model `{model_id}` is not available on the requested runtime"
                    ),
                });
            }
            if !matches!(slot.state, ModelSlotState::Loaded(_)) {
                return Err(Error::ModelNotLoaded {
                    model_id: model_id.clone(),
                });
            }
            return Ok(slot.descriptor.clone());
        }

        let loaded = self
            .slots
            .values()
            .filter(|slot| matches_runtime_preference(&slot.state, selection.runtime_preference))
            .map(|slot| slot.descriptor.clone())
            .collect::<Vec<_>>();

        if loaded.is_empty() {
            return Err(Error::UnresolvedModelSelection {
                message: "no loaded model satisfies the requested selection".to_string(),
            });
        }

        let registry = StaticModelRegistry::new(loaded)?;
        registry
            .resolve_model(selection)
            .ok_or_else(|| Error::UnresolvedModelSelection {
                message: "no loaded model satisfies the requested selection".to_string(),
            })
    }

    fn slot_mut(&mut self, model_id: &ModelId) -> Result<&mut ModelSlot> {
        self.slots
            .get_mut(model_id)
            .ok_or_else(|| Error::UnknownModel {
                model_id: model_id.clone(),
            })
    }

    async fn build_runtime(
        runtime_config: &RuntimeConfig,
        model: &RegisteredModelConfig,
        runtime_kind: InferenceRuntime,
    ) -> Result<BackendRuntime> {
        match runtime_kind {
            InferenceRuntime::AnyTts => match internal_runtime_kind(model) {
                Some(_) => Ok(BackendRuntime::AnyTts(
                    AnyTtsRuntime::from_model_config(model).await?,
                )),
                None => Err(Error::UnsupportedFeature {
                    feature: "runtime selection must resolve to a concrete runtime".to_string(),
                }),
            },
            InferenceRuntime::MistralRs => Ok(BackendRuntime::MistralRs(
                MistralRuntime::from_model_config(runtime_config, model).await?,
            )),
            InferenceRuntime::Mold => Ok(BackendRuntime::Mold(
                MoldRuntime::from_model_config(runtime_config, model).await?,
            )),
        }
    }

    fn resolve_load_runtime_kind(
        model: &RegisteredModelConfig,
        requested_runtime_kind: InferenceRuntime,
    ) -> Result<InferenceRuntime> {
        if requested_runtime_kind != InferenceRuntime::AnyTts {
            return model
                .supports_runtime(requested_runtime_kind)
                .then_some(requested_runtime_kind)
                .ok_or_else(|| Error::UnsupportedFeature {
                    feature: format!(
                        "model `{}` is not configured for runtime `{requested_runtime_kind:?}`",
                        model.descriptor.id
                    ),
                });
        }

        if model.descriptor.runtime != InferenceRuntime::AnyTts {
            return Ok(model.descriptor.runtime);
        }

        if internal_runtime_kind(model).is_some() {
            return Ok(InferenceRuntime::AnyTts);
        }

        let mut runtimes = model.runtimes.iter().map(|runtime| runtime.runtime());
        let Some(first_runtime) = runtimes.next() else {
            return Err(Error::UnsupportedFeature {
                feature: format!(
                    "model `{}` does not expose any loadable runtime",
                    model.descriptor.id
                ),
            });
        };

        let _ = runtimes;
        Ok(first_runtime)
    }
}

fn request_model_selection(request: &InferenceRequest) -> &ModelSelection {
    match request {
        InferenceRequest::Generate(request) => &request.model,
        InferenceRequest::Embed(request) => &request.model,
        InferenceRequest::GenerateImage(request) => &request.model,
        InferenceRequest::GenerateSpeech(request) => &request.model,
        InferenceRequest::Tokenize(request) => &request.model,
        InferenceRequest::Detokenize(request) => &request.model,
    }
}

fn request_session_id(request: &InferenceRequest) -> Option<&str> {
    match request {
        InferenceRequest::Generate(GenerateRequest { session_id, .. }) => {
            session_id.as_ref().map(|session_id| session_id.as_str())
        }
        _ => None,
    }
}

fn requested_runtime_kind(request: &InferenceRequest) -> Option<InferenceRuntime> {
    (request_model_selection(request).runtime_preference != InferenceRuntime::AnyTts)
        .then_some(request_model_selection(request).runtime_preference)
}

fn should_switch_runtime(
    requested_runtime_kind: Option<InferenceRuntime>,
    active_runtime_kind: InferenceRuntime,
) -> bool {
    requested_runtime_kind.is_some_and(|runtime_kind| runtime_kind != active_runtime_kind)
}

fn matches_runtime_preference(
    state: &ModelSlotState,
    requested_runtime_kind: InferenceRuntime,
) -> bool {
    match requested_runtime_kind {
        InferenceRuntime::AnyTts => matches!(state, ModelSlotState::Loaded(_)),
        runtime_kind => matches!(
            state,
            ModelSlotState::Loaded(ActiveModelRuntime {
                runtime_kind: active_runtime_kind,
                ..
            }) if *active_runtime_kind == runtime_kind
        ),
    }
}

fn should_reload_for_session(request: &InferenceRequest, active_session_id: Option<&str>) -> bool {
    match request {
        InferenceRequest::Generate(_) => {
            let requested_session_id = request_session_id(request);
            match active_session_id {
                Some(active_session_id) => requested_session_id != Some(active_session_id),
                None => false,
            }
        }
        _ => false,
    }
}

fn supports_selection(descriptor: &ModelDescriptor, selection: &ModelSelection) -> bool {
    selection
        .required_capabilities
        .iter()
        .all(|capability| descriptor.capabilities.contains(capability))
}

#[cfg(test)]
mod tests {
    use nexo_core::{
        Conversation, MetadataMap, ModelCapability, ModelModalities, RoleStrategy, SessionId,
        StreamingMode, SupportedModality, ToolChoice,
    };

    use super::*;

    #[test]
    fn generate_request_reloads_when_session_changes() {
        let request = generate_request(Some("session-b"));
        assert!(should_reload_for_session(&request, Some("session-a")));
        assert!(!should_reload_for_session(&request, Some("session-b")));
    }

    #[test]
    fn generate_request_reloads_when_session_presence_changes() {
        assert!(should_reload_for_session(
            &generate_request(None),
            Some("session-a")
        ));
        assert!(!should_reload_for_session(
            &generate_request(Some("session-a")),
            None,
        ));
    }

    #[test]
    fn generate_request_does_not_reload_before_first_session_is_bound() {
        assert!(!should_reload_for_session(
            &generate_request(Some("session-a")),
            None,
        ));
        assert!(!should_reload_for_session(&generate_request(None), None));
    }

    fn generate_request(session_id: Option<&str>) -> InferenceRequest {
        InferenceRequest::Generate(nexo_core::inference::request::GenerateRequest {
            request_id: None,
            session_id: session_id.map(SessionId::from),
            run_id: None,
            round_id: None,
            model: nexo_core::ModelSelection {
                specific_model: Some(ModelId::from("chat")),
                required_capabilities: vec![ModelCapability::TextGeneration],
                preferred_capabilities: Vec::new(),
                runtime_preference: Default::default(),
            },
            conversation: Conversation {
                messages: Vec::new(),
                metadata: MetadataMap::new(),
            },
            tools: Vec::new(),
            tool_choice: ToolChoice::Disabled,
            reasoning: Default::default(),
            output_constraint: nexo_core::OutputConstraint::None,
            sampling: Default::default(),
            streaming: StreamingMode::Buffered,
            metadata: MetadataMap::new(),
        })
    }

    #[test]
    fn load_model_resolves_any_for_single_runtime_model() {
        let model = RegisteredModelConfig {
            descriptor: test_descriptor("gemma"),
            runtimes: vec![crate::ModelRuntimeImplementation::MistralRs(
                crate::engine::mistralrs::MistralRsModelConfig {
                    loader: crate::engine::mistralrs::MistralRsLoader::Auto(
                        crate::engine::mistralrs::MistralRsAutoLoader {
                            model_id: "gemma".to_string(),
                            from_uqff: None,
                            tokenizer_json: None,
                            chat_template: None,
                            jinja_explicit: None,
                            dtype: crate::ModelDataType::Auto,
                            hf_cache_path: None,
                        },
                    ),
                    revision: None,
                },
            )],
        };

        assert_eq!(
            RuntimeManager::resolve_load_runtime_kind(&model, InferenceRuntime::AnyTts).unwrap(),
            InferenceRuntime::MistralRs
        );
    }

    #[test]
    fn load_model_keeps_internal_runtime_private_behind_any() {
        let mut descriptor = test_descriptor("kokoro-82m-tts");
        descriptor.metadata.insert(
            crate::engine::any_tts::INTERNAL_RUNTIME_KEY.to_string(),
            serde_json::Value::String(crate::engine::any_tts::KOKORO_RUNTIME_ID.to_string()),
        );
        let model = RegisteredModelConfig {
            descriptor,
            runtimes: Vec::new(),
        };

        assert_eq!(
            RuntimeManager::resolve_load_runtime_kind(&model, InferenceRuntime::AnyTts).unwrap(),
            InferenceRuntime::AnyTts
        );
    }

    fn test_descriptor(id: &str) -> ModelDescriptor {
        ModelDescriptor {
            id: id.into(),
            display_name: id.to_string(),
            provider: Some("test".to_string()),
            runtime: InferenceRuntime::AnyTts,
            capabilities: vec![ModelCapability::TextGeneration],
            modalities: ModelModalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            role_strategy: RoleStrategy::Default,
            context_window_tokens: Some(4096),
            max_output_tokens: Some(1024),
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn supports_selection_checks_required_capabilities() {
        let descriptor = ModelDescriptor {
            id: ModelId::from("chat"),
            display_name: "chat".to_string(),
            provider: Some("test".to_string()),
            runtime: InferenceRuntime::AnyTts,
            capabilities: vec![ModelCapability::TextGeneration],
            modalities: ModelModalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            role_strategy: RoleStrategy::Default,
            context_window_tokens: Some(4096),
            max_output_tokens: Some(1024),
            metadata: MetadataMap::new(),
        };
        let selection = ModelSelection {
            specific_model: Some(ModelId::from("chat")),
            required_capabilities: vec![ModelCapability::TextGeneration],
            preferred_capabilities: Vec::new(),
            runtime_preference: Default::default(),
        };

        assert!(supports_selection(&descriptor, &selection));
    }
}
