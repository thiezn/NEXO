use crate::server::state::{GatewayState, PeerId, SharedState};
use nexo_core::{ModelCapability, ModelDefinition};
use nexo_ws_schema::{
    ConnectionRole, Frame, Method, ModelLoadParams, ModelLoadResponse, ModelUnloadParams,
};
use tokio::sync::mpsc;

use super::inference::InferenceOutcome;

/// Timeout for model load operations.
const MODEL_LOAD_TIMEOUT_SECS: u64 = 300;

/// Stateless inference node selection and model preparation.
pub(crate) struct Router;

#[derive(Debug)]
pub(crate) enum RouteError {
    NoCapableNode,
    Error(String),
}

impl Router {
    /// Select a node for inference or return a terminal inference outcome when routing cannot proceed.
    pub(super) async fn route_inference(
        state: &SharedState,
        model_id: Option<&str>,
    ) -> Result<(PeerId, mpsc::Sender<Frame>), InferenceOutcome> {
        match model_id {
            Some(model_id) => {
                let selected = {
                    let state_read = state.read().await;
                    Self::find_loaded_model(&state_read, model_id)
                        .map(RoutedPeer::Ready)
                        .or_else(|| {
                            Self::find_available_model(&state_read, model_id)
                                .map(RoutedPeer::NeedsModelLoad)
                        })
                };

                match selected {
                    Some(RoutedPeer::Ready(selection)) => Ok(selection),
                    Some(RoutedPeer::NeedsModelLoad((peer_id, node_sender))) => {
                        Self::ensure_model_loaded(model_id, peer_id, node_sender, state).await
                    }
                    Some(RoutedPeer::NeedsCapabilityLoad { .. }) => Err(InferenceOutcome::Error(
                        "Unexpected capability route during explicit model routing".into(),
                    )),
                    None => Err(InferenceOutcome::NoLlmAvailable),
                }
            }
            None => {
                let selected = {
                    let state_read = state.read().await;
                    Self::find_loaded_llm(&state_read)
                        .map(RoutedPeer::Ready)
                        .or_else(|| {
                            Self::find_available_capability(
                                &state_read,
                                ModelCapability::TextGeneration,
                            )
                            .map(|(model_id, peer_id, sender)| {
                                RoutedPeer::NeedsCapabilityLoad {
                                    model_id,
                                    peer_id,
                                    sender,
                                }
                            })
                        })
                };

                match selected {
                    Some(RoutedPeer::Ready(selection)) => Ok(selection),
                    Some(RoutedPeer::NeedsCapabilityLoad {
                        model_id,
                        peer_id,
                        sender,
                    }) => Self::ensure_model_loaded(&model_id, peer_id, sender, state).await,
                    Some(RoutedPeer::NeedsModelLoad(_)) => Err(InferenceOutcome::Error(
                        "Unexpected explicit model route during default LLM routing".into(),
                    )),
                    None => Err(InferenceOutcome::NoLlmAvailable),
                }
            }
        }
    }

    /// Select a node that can satisfy a model capability, loading that model first when needed.
    pub(crate) async fn route_capability(
        state: &SharedState,
        capability: ModelCapability,
    ) -> Result<(PeerId, mpsc::Sender<Frame>), RouteError> {
        let selected = {
            let state_read = state.read().await;
            Self::find_loaded_capability(&state_read, capability)
                .map(RoutedPeer::Ready)
                .or_else(|| {
                    Self::find_available_capability(&state_read, capability).map(
                        |(model_id, peer_id, sender)| RoutedPeer::NeedsCapabilityLoad {
                            model_id,
                            peer_id,
                            sender,
                        },
                    )
                })
        };

        match selected {
            Some(RoutedPeer::Ready(selection)) => Ok(selection),
            Some(RoutedPeer::NeedsCapabilityLoad {
                model_id,
                peer_id,
                sender,
            }) => Self::ensure_model_loaded(&model_id, peer_id, sender, state)
                .await
                .map_err(RouteError::from),
            Some(RoutedPeer::NeedsModelLoad(_)) => Err(RouteError::Error(
                "Unexpected explicit model route during capability routing".into(),
            )),
            None => Err(RouteError::NoCapableNode),
        }
    }

    fn find_loaded_model(
        state: &GatewayState,
        model_id: &str,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        Self::find_node_peer(state, |peer_id| {
            state
                .loaded_models
                .get(peer_id)
                .is_some_and(|models| models.iter().any(|model| model.id.as_str() == model_id))
        })
    }

    fn find_available_model(
        state: &GatewayState,
        model_id: &str,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        Self::find_node_peer(state, |peer_id| {
            state.available_models.get(peer_id).is_some_and(|models| {
                models
                    .iter()
                    .any(|available_model| available_model == model_id)
            })
        })
    }

    fn find_loaded_llm(state: &GatewayState) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        Self::find_node_peer(state, |peer_id| {
            state.loaded_models.get(peer_id).is_some_and(|models| {
                models.iter().any(|model| {
                    model.capabilities.iter().copied().any(|capability| {
                        matches!(
                            capability,
                            ModelCapability::TextGeneration | ModelCapability::ToolCalling
                        )
                    })
                })
            })
        })
    }

    fn find_loaded_capability(
        state: &GatewayState,
        capability: ModelCapability,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        Self::find_node_peer(state, |peer_id| {
            state.loaded_models.get(peer_id).is_some_and(|models| {
                models
                    .iter()
                    .any(|model| Self::supports_capability(model, capability))
            })
        })
    }

    fn find_available_capability(
        state: &GatewayState,
        capability: ModelCapability,
    ) -> Option<(String, PeerId, mpsc::Sender<Frame>)> {
        state.peers.iter().find_map(|(peer_id, peer)| {
            if peer.role != ConnectionRole::Node {
                return None;
            }

            let model_id = state
                .available_model_descriptors
                .get(peer_id)?
                .iter()
                .find(|model| Self::supports_capability(model, capability))?
                .id
                .to_string();

            state
                .peer_senders
                .get(peer_id)
                .cloned()
                .map(|sender| (model_id, peer_id.clone(), sender))
        })
    }

    fn supports_capability(model: &ModelDefinition, capability: ModelCapability) -> bool {
        model.capabilities.contains(&capability)
    }

    fn find_node_peer(
        state: &GatewayState,
        mut matches: impl FnMut(&PeerId) -> bool,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        state.peers.iter().find_map(|(peer_id, peer)| {
            if peer.role != ConnectionRole::Node || !matches(peer_id) {
                return None;
            }

            state
                .peer_senders
                .get(peer_id)
                .cloned()
                .map(|sender| (peer_id.clone(), sender))
        })
    }

    async fn ensure_model_loaded(
        model_id: &str,
        peer_id: PeerId,
        node_sender: mpsc::Sender<Frame>,
        state: &SharedState,
    ) -> Result<(PeerId, mpsc::Sender<Frame>), InferenceOutcome> {
        let models_to_unload: Vec<String> = state
            .read()
            .await
            .loaded_models
            .get(&peer_id)
            .map(|models| {
                models
                    .iter()
                    .filter(|model| model.id.as_str() != model_id)
                    .map(|model| model.id.to_string())
                    .collect()
            })
            .unwrap_or_default();

        for old_model in &models_to_unload {
            let unload_params = ModelUnloadParams {
                model_id: old_model.clone(),
            };
            let unload_request_id = Frame::new_id();
            let frame = Frame::Request {
                id: unload_request_id.clone(),
                method: Method::ModelUnload,
                params: serde_json::to_value(&unload_params).unwrap_or_default(),
            };
            let (tx, rx) = tokio::sync::oneshot::channel();
            state
                .write()
                .await
                .pending_requests
                .insert(unload_request_id.clone(), tx);
            if node_sender.send(frame).await.is_err() {
                state
                    .write()
                    .await
                    .pending_requests
                    .remove(&unload_request_id);
                tracing::error!(
                    peer_id,
                    model_id = old_model,
                    "Failed to send ModelUnload request to node"
                );
            } else {
                match tokio::time::timeout(std::time::Duration::from_secs(10), rx).await {
                    Ok(Ok(Frame::Response { ok: true, .. })) => {
                        tracing::info!(peer_id, model_id = old_model, "Model unloaded");
                    }
                    Ok(Ok(Frame::Response {
                        ok: false, error, ..
                    })) => {
                        let message = error
                            .map(|payload| payload.message)
                            .unwrap_or_else(|| "ModelUnload failed without error payload".into());
                        tracing::error!(
                            peer_id,
                            model_id = old_model,
                            error = %message,
                            "Node failed to unload model"
                        );
                    }
                    Ok(Ok(other)) => {
                        tracing::error!(
                            peer_id,
                            model_id = old_model,
                            frame = ?other,
                            "Unexpected frame type from node during model unload"
                        );
                    }
                    Ok(Err(_)) => {
                        tracing::error!(
                            peer_id,
                            model_id = old_model,
                            "Node disconnected during model unload"
                        );
                    }
                    Err(_) => {
                        state
                            .write()
                            .await
                            .pending_requests
                            .remove(&unload_request_id);
                        tracing::error!(peer_id, model_id = old_model, "Model unload timed out");
                    }
                }
            }
        }

        if !models_to_unload.is_empty() {
            state.write().await.set_loaded_models(&peer_id, Vec::new());
        }

        let load_params = ModelLoadParams {
            model_id: model_id.to_string(),
        };
        let load_request_id = Frame::new_id();
        let frame = Frame::Request {
            id: load_request_id.clone(),
            method: Method::ModelLoad,
            params: serde_json::to_value(&load_params).unwrap_or_default(),
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        state
            .write()
            .await
            .pending_requests
            .insert(load_request_id.clone(), response_tx);

        if node_sender.send(frame).await.is_err() {
            state
                .write()
                .await
                .pending_requests
                .remove(&load_request_id);
            tracing::error!(
                peer_id,
                model_id,
                "Failed to send ModelLoad request to node"
            );
            return Err(InferenceOutcome::Error(format!(
                "Failed to send ModelLoad request to node {peer_id}"
            )));
        }

        match tokio::time::timeout(
            std::time::Duration::from_secs(MODEL_LOAD_TIMEOUT_SECS),
            response_rx,
        )
        .await
        {
            Ok(Ok(Frame::Response {
                ok: true, payload, ..
            })) => {
                let response = payload.as_ref().and_then(|payload| {
                    serde_json::from_value::<ModelLoadResponse>(payload.clone()).ok()
                });
                if response.as_ref().is_none_or(|response| response.loaded) {
                    tracing::info!(peer_id, model_id, "Model loaded on node");
                    Ok((peer_id, node_sender))
                } else {
                    let message =
                        response
                            .and_then(|response| response.error)
                            .unwrap_or_else(|| {
                                format!("Node {peer_id} failed to load model '{model_id}'")
                            });
                    tracing::error!(peer_id, model_id, error = %message, "Node failed to load model");
                    Err(InferenceOutcome::Error(format!(
                        "Node {peer_id} failed to load model '{model_id}': {message}"
                    )))
                }
            }
            Ok(Ok(Frame::Response {
                ok: false, error, ..
            })) => {
                let message = error
                    .map(|payload| format!("ModelLoad error: {}", payload.message))
                    .unwrap_or_else(|| format!("ModelLoad failed on node {peer_id}"));
                tracing::error!(peer_id, model_id, error = %message, "Node rejected model load");
                Err(InferenceOutcome::Error(message))
            }
            Ok(Ok(other)) => {
                tracing::error!(peer_id, model_id, frame = ?other, "Unexpected frame type from node during model load");
                Err(InferenceOutcome::Error(
                    "Unexpected frame type from node during model load".into(),
                ))
            }
            Ok(Err(_)) => {
                tracing::error!(peer_id, model_id, "Node disconnected during model load");
                Err(InferenceOutcome::Error(
                    "Node disconnected during model load".into(),
                ))
            }
            Err(_) => {
                state
                    .write()
                    .await
                    .pending_requests
                    .remove(&load_request_id);
                tracing::error!(
                    peer_id,
                    model_id,
                    timeout_secs = MODEL_LOAD_TIMEOUT_SECS,
                    "Model load timed out"
                );
                Err(InferenceOutcome::Error(format!(
                    "Model load timed out after {MODEL_LOAD_TIMEOUT_SECS}s"
                )))
            }
        }
    }
}

enum RoutedPeer {
    Ready((PeerId, mpsc::Sender<Frame>)),
    NeedsModelLoad((PeerId, mpsc::Sender<Frame>)),
    NeedsCapabilityLoad {
        model_id: String,
        peer_id: PeerId,
        sender: mpsc::Sender<Frame>,
    },
}

impl From<InferenceOutcome> for RouteError {
    fn from(outcome: InferenceOutcome) -> Self {
        match outcome {
            InferenceOutcome::NoLlmAvailable => Self::NoCapableNode,
            InferenceOutcome::Error(error) => Self::Error(error),
            InferenceOutcome::Reply(_) | InferenceOutcome::ToolCalls(_) => {
                Self::Error("Unexpected inference outcome during routing".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::server::state::PeerInfo;
    use nexo_core::{MetadataMap, ModelId, RoleStrategy};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_loaded_model(
        model_id: &str,
        capabilities: Vec<ModelCapability>,
    ) -> nexo_core::ModelDefinition {
        nexo_core::ModelDefinition {
            id: ModelId::from(model_id),
            display_name: model_id.into(),
            provider: Some("test".into()),
            runtime: nexo_core::InferenceRuntime::AnyTts,
            capabilities,
            role_strategy: RoleStrategy::Default,
            context_window_tokens: Some(4096),
            max_output_tokens: Some(1024),
            metadata: MetadataMap::new(),
        }
    }

    fn make_node_peer(id: &str) -> PeerInfo {
        PeerInfo {
            id: id.into(),
            client_id: "rust-node".into(),
            role: ConnectionRole::Node,
            scopes: vec![],
            capabilities: vec![],
            commands: vec![],
            device_id: Some("dev-2".into()),
            connected_at: chrono::Utc::now(),
        }
    }

    fn shared_state() -> SharedState {
        Arc::new(RwLock::new(GatewayState::new(PathBuf::from("/tmp"))))
    }

    #[tokio::test]
    async fn route_inference_prefers_loaded_model() {
        let state = shared_state();
        let (node_tx, _node_rx) = mpsc::channel(4);
        {
            let mut state_write = state.write().await;
            state_write.add_peer(make_node_peer("n1"), node_tx);
            state_write.set_loaded_models(
                "n1",
                vec![make_loaded_model(
                    "gemma-3n",
                    vec![ModelCapability::TextGeneration],
                )],
            );
        }

        let (peer_id, _sender) = match Router::route_inference(&state, Some("gemma-3n")).await {
            Ok(selection) => selection,
            Err(_) => panic!("expected loaded model route"),
        };

        assert_eq!(peer_id, "n1");
    }

    #[tokio::test]
    async fn route_inference_loads_available_model() {
        let state = shared_state();
        let (node_tx, mut node_rx) = mpsc::channel(4);
        {
            let mut state_write = state.write().await;
            state_write.add_peer(make_node_peer("n1"), node_tx);
            state_write.set_available_models("n1", vec!["chat-b".into()]);
        }

        let state_for_response = state.clone();
        tokio::spawn(async move {
            let frame = node_rx.recv().await.unwrap();
            let request_id = match frame {
                Frame::Request { id, method, .. } => {
                    assert_eq!(method, Method::ModelLoad);
                    id
                }
                _ => panic!("expected model load request"),
            };

            let response_tx = {
                let mut state_write = state_for_response.write().await;
                state_write.pending_requests.remove(&request_id).unwrap()
            };

            response_tx
                .send(Frame::Response {
                    id: request_id,
                    ok: true,
                    payload: Some(
                        serde_json::to_value(ModelLoadResponse {
                            model_id: "chat-b".into(),
                            loaded: true,
                            error: None,
                        })
                        .unwrap(),
                    ),
                    error: None,
                })
                .unwrap();
        });

        let (peer_id, _sender) = match Router::route_inference(&state, Some("chat-b")).await {
            Ok(selection) => selection,
            Err(_) => panic!("expected loadable model route"),
        };

        assert_eq!(peer_id, "n1");
    }

    #[tokio::test]
    async fn route_inference_without_explicit_model_uses_loaded_llm() {
        let state = shared_state();
        let (chat_tx, _chat_rx) = mpsc::channel(4);
        let (image_tx, _image_rx) = mpsc::channel(4);
        {
            let mut state_write = state.write().await;
            state_write.add_peer(make_node_peer("n-chat"), chat_tx);
            state_write.add_peer(make_node_peer("n-image"), image_tx);
            state_write.set_loaded_models(
                "n-chat",
                vec![make_loaded_model(
                    "chatty",
                    vec![
                        ModelCapability::ToolCalling,
                        ModelCapability::TextGeneration,
                    ],
                )],
            );
            state_write.set_loaded_models(
                "n-image",
                vec![make_loaded_model(
                    "vision",
                    vec![ModelCapability::ImageInput],
                )],
            );
        }

        let (peer_id, _sender) = match Router::route_inference(&state, None).await {
            Ok(selection) => selection,
            Err(_) => panic!("expected default llm route"),
        };

        assert_eq!(peer_id, "n-chat");
    }

    #[tokio::test]
    async fn route_inference_without_explicit_model_loads_available_llm() {
        let state = shared_state();
        let (node_tx, mut node_rx) = mpsc::channel(4);
        {
            let mut state_write = state.write().await;
            state_write.add_peer(make_node_peer("n1"), node_tx);
            state_write.set_loaded_models(
                "n1",
                vec![make_loaded_model(
                    "image-gen",
                    vec![ModelCapability::ImageGeneration],
                )],
            );
            state_write.set_available_model_descriptors(
                "n1",
                vec![
                    make_loaded_model("image-gen", vec![ModelCapability::ImageGeneration]),
                    make_loaded_model("chatty", vec![ModelCapability::TextGeneration]),
                ],
            );
        }

        let state_for_response = state.clone();
        tokio::spawn(async move {
            let unload_id = match node_rx.recv().await.unwrap() {
                Frame::Request { id, method, params } => {
                    assert_eq!(method, Method::ModelUnload);
                    assert_eq!(params["modelId"], "image-gen");
                    id
                }
                _ => panic!("expected model unload request"),
            };

            let unload_tx = {
                let mut state_write = state_for_response.write().await;
                state_write.pending_requests.remove(&unload_id).unwrap()
            };
            unload_tx
                .send(Frame::Response {
                    id: unload_id,
                    ok: true,
                    payload: Some(
                        serde_json::to_value(nexo_ws_schema::ModelUnloadResponse {
                            unloaded: true,
                        })
                        .unwrap(),
                    ),
                    error: None,
                })
                .unwrap();

            let load_id = match node_rx.recv().await.unwrap() {
                Frame::Request { id, method, params } => {
                    assert_eq!(method, Method::ModelLoad);
                    assert_eq!(params["modelId"], "chatty");
                    id
                }
                _ => panic!("expected model load request"),
            };

            let load_tx = {
                let mut state_write = state_for_response.write().await;
                state_write.pending_requests.remove(&load_id).unwrap()
            };
            load_tx
                .send(Frame::Response {
                    id: load_id,
                    ok: true,
                    payload: Some(
                        serde_json::to_value(ModelLoadResponse {
                            model_id: "chatty".into(),
                            loaded: true,
                            error: None,
                        })
                        .unwrap(),
                    ),
                    error: None,
                })
                .unwrap();
        });

        let (peer_id, _sender) = match Router::route_inference(&state, None).await {
            Ok(selection) => selection,
            Err(_) => panic!("expected loadable default llm route"),
        };

        assert_eq!(peer_id, "n1");
    }

    #[tokio::test]
    async fn route_inference_returns_no_llm_available() {
        let state = shared_state();

        let outcome = Router::route_inference(&state, None).await.unwrap_err();

        assert!(matches!(outcome, InferenceOutcome::NoLlmAvailable));
    }
}
