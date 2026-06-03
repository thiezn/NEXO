//! Persistence helpers for conversations, prompts, runs, and sessions.

mod conversations;
mod db_types;
mod prompts;
mod runs;
mod sessions;
#[cfg(test)]
mod tests;

pub use conversations::{
    append_run_instructions, insert_conversation_entry, load_conversation_messages,
};
pub use db_types::{
    ConversationEntryKind, RoundStatus, RunSummaryKind, ToolTraceStatus, run_status_to_db,
};
pub use prompts::{
    create_prompt_document, delete_prompt_collection, delete_prompt_document,
    list_prompt_collections, list_prompt_documents, upsert_prompt_collection,
};
pub use runs::{
    create_round, create_run, create_run_with_tool_choice, create_tool_trace, finish_round,
    finish_run, finish_tool_trace, is_run_cancelled, mark_run_queued, next_round_index,
    stop_run,
};
pub use sessions::{clear_session, create_session, get_session, list_sessions};

#[cfg(test)]
pub(crate) use conversations::insert_message;
#[cfg(test)]
pub(crate) use runs::store_run_summary;

pub(crate) use runs::decode_reasoning_json;
pub(crate) use runs::decode_tool_choice_json;

pub(crate) use prompts::load_prompt_collection_system_prompt;
