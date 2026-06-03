//! Run-context assembly for model inference rounds.

use crate::{agent::persistence, server::state::SharedState};
use nexo_core::{ContentPart, ConversationMessage, MessageRole, MetadataMap, TextPart};
use nexo_ws_schema::SystemPrompt;
use sqlx::SqlitePool;

/// Prepared context for a single inference round.
pub(crate) struct PreparedContext {
    pub persisted_message_count: usize,
    pub round_messages: Vec<ConversationMessage>,
}

/// Builds the model-facing context for each run round.
pub(crate) struct ContextManager {
    system_prompt: Option<SystemPrompt>,
}

impl ContextManager {
    /// Load prompt-backed configuration needed for the lifetime of a run.
    pub(crate) async fn new(state: &SharedState, prompt_collection_id: Option<&str>) -> Self {
        let system_prompt = load_system_prompt(state, prompt_collection_id).await;
        Self { system_prompt }
    }

    /// Assemble the messages sent to the model for the next round.
    pub(crate) async fn prepare_round_context(
        &self,
        db: &SqlitePool,
        session_id: &str,
    ) -> Result<PreparedContext, sqlx::Error> {
        let conversation_messages = persistence::load_conversation_messages(db, session_id).await?;
        let round_messages = assemble_round_messages(&conversation_messages, self.system_prompt.as_ref());

        Ok(PreparedContext {
            persisted_message_count: conversation_messages.len(),
            round_messages,
        })
    }
}

async fn load_system_prompt(
    state: &SharedState,
    collection_id: Option<&str>,
) -> Option<SystemPrompt> {
    let git = state.read().await.git_storage.clone();
    if let Some(git) = git {
        let selected_collection_id = collection_id.map(str::to_owned);
        tokio::task::spawn_blocking(move || {
            selected_collection_id.and_then(|collection_id| {
                persistence::load_prompt_collection_system_prompt(&git, &collection_id)
                    .ok()
                    .flatten()
            })
        })
        .await
        .unwrap_or(None)
    } else {
        None
    }
}

fn assemble_round_messages(
    conversation_messages: &[ConversationMessage],
    system_prompt: Option<&SystemPrompt>,
) -> Vec<ConversationMessage> {
    let mut system_parts = Vec::new();
    if let Some(system_prompt) = system_prompt
        && !system_prompt.content.is_empty()
    {
        system_parts.push(system_prompt.content.clone());
    }

    let system_prompt = if system_parts.is_empty() {
        super::engine::DEFAULT_SYSTEM_PROMPT.to_string()
    } else {
        system_parts.join("\n\n")
    };

    let system_message = ConversationMessage {
        role: MessageRole::System,
        parts: vec![ContentPart::Text(TextPart {
            text: system_prompt,
        })],
        metadata: MetadataMap::new(),
    };

    std::iter::once(system_message)
        .chain(conversation_messages.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_round_messages_does_not_render_tool_catalog() {
        let messages = assemble_round_messages(&[], None);
        let Some(ConversationMessage { parts, .. }) = messages.first() else {
            panic!("expected system message");
        };

        let Some(ContentPart::Text(TextPart { text })) = parts.first() else {
            panic!("expected text system prompt");
        };

        assert!(!text.contains("# Available Tools"));
    }
}
