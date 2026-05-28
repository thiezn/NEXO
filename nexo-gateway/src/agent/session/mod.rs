//! Session, conversation, and run persistence helpers for the run runtime.

mod conversation;
mod runs;
mod sessions;
#[cfg(test)]
mod tests;

pub use conversation::{
    ENTRY_ASSISTANT_RESPONSE, ENTRY_INSTRUCTION, ENTRY_TOOL_CALL_INTENT, ENTRY_TOOL_RESULT,
    ENTRY_USER_INPUT, append_run_context, insert_conversation_entry,
};
pub use runs::{
    create_round, create_run, create_tool_trace, finish_round, finish_run, finish_tool_trace,
    is_run_cancelled, next_round_index, stop_run,
};
pub use sessions::{clear_session, create_session, get_session, list_sessions};

#[cfg(test)]
pub(crate) use conversation::insert_message;
#[cfg(test)]
pub(crate) use runs::store_run_summary;
