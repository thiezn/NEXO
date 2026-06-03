use std::collections::{BTreeMap, BTreeSet};

use nexo_core::inference::request::GenerateRequest;
use nexo_core::{InferenceRequest, ModelDescriptor, ModelId, ModelSelection};

use crate::{
    Error, NexoAiConfig, RegisteredModelConfig, Result, RuntimeConfig, StaticModelRegistry,
};

use super::engine::MistralRuntime;

#[derive(Debug)]
pub(crate) struct RuntimeController {
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
    Loaded(LoadedModelRuntime),
}

#[derive(Debug, Clone)]
struct LoadedModelRuntime {
    runtime: MistralRuntime,
    active_session_id: Option<String>,
}

impl RuntimeController {
    pub(crate) fn new(config: &NexoAiConfig) -> Self {
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

    pub(crate) async fn load_model(&mut self, model_id: &ModelId) -> Result<()> {
        let runtime_config = self.runtime_config.clone();
        let config = match self.slots.get(model_id) {
            Some(slot) if matches!(slot.state, ModelSlotState::Loaded(_)) => return Ok(()),
            Some(slot) => slot.config.clone(),
            None => {
                return Err(Error::UnknownModel {
                    model_id: model_id.clone(),
                });
            }
        };

        let runtime = Self::build_runtime(&runtime_config, &config).await?;
        let slot = self.slot_mut(model_id)?;
        slot.state = ModelSlotState::Loaded(LoadedModelRuntime {
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
    ) -> Result<(ModelDescriptor, MistralRuntime)> {
        let descriptor = self.resolve_loaded_model(request)?;
        let model_id = descriptor.id.clone();
        let requested_session_id = request_session_id(request).map(str::to_owned);
        let runtime_config = self.runtime_config.clone();

        let (config, state) = {
            let slot = self.slot_mut(&model_id)?;
            let state = std::mem::replace(&mut slot.state, ModelSlotState::Unloaded);
            (slot.config.clone(), state)
        };

        let ModelSlotState::Loaded(mut loaded) = state else {
            return Err(Error::ModelNotLoaded { model_id });
        };

        let runtime = if should_reload_for_session(request, loaded.active_session_id.as_deref()) {
            tracing::info!(
                model_id = %model_id,
                previous_session_id = ?loaded.active_session_id,
                requested_session_id = ?requested_session_id,
                "Reloading model runtime for session change"
            );
            drop(loaded);
            Self::build_runtime(&runtime_config, &config).await?
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
        slot.state = ModelSlotState::Loaded(LoadedModelRuntime {
            runtime: runtime.clone(),
            active_session_id: requested_session_id,
        });
        Ok((descriptor, runtime))
    }

    fn resolve_loaded_model(&self, request: &InferenceRequest) -> Result<ModelDescriptor> {
        let selection = match request {
            InferenceRequest::Generate(request) => &request.model,
            InferenceRequest::Embed(request) => &request.model,
            InferenceRequest::GenerateImage(request) => &request.model,
            InferenceRequest::GenerateSpeech(request) => &request.model,
            InferenceRequest::Tokenize(request) => &request.model,
            InferenceRequest::Detokenize(request) => &request.model,
        };

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
            .filter(|slot| matches!(slot.state, ModelSlotState::Loaded(_)))
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
    ) -> Result<MistralRuntime> {
        MistralRuntime::from_model_config(runtime_config, model).await
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
    fn supports_selection_checks_required_capabilities() {
        let descriptor = ModelDescriptor {
            id: ModelId::from("chat"),
            display_name: "chat".to_string(),
            provider: Some("test".to_string()),
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
        };

        assert!(supports_selection(&descriptor, &selection));
    }
}
