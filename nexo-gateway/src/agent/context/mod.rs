//! Conversation context loading and stored prompt-asset management.

mod collections;
mod conversation;
mod documents;
mod prompt;
mod records;

pub use collections::{
    delete_context_collection, list_context_collections, upsert_context_collection,
};
pub use conversation::{ConversationContextMessage, load_conversation_context};
pub use documents::{
    create_context_document, delete_context_document, list_context_documents, read_context_document,
};
pub use prompt::{SystemPromptAssets, build_tool_prompt_section, load_system_prompt_assets};
pub use records::{ContextCollection, ContextDocument};
