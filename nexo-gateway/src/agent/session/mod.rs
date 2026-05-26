//! Session, transcript, and run persistence helpers for the run runtime.

mod runs;
mod sessions;
#[cfg(test)]
mod tests;
mod transcript;

pub use runs::{
    create_round, create_run, create_tool_trace, finish_round, finish_run, finish_tool_trace,
    is_run_cancelled, next_round_index, stop_run,
};
pub use sessions::{clear_session, create_session, get_session, list_sessions};
pub use transcript::{append_run_context, insert_transcript_entry};

#[cfg(test)]
pub(crate) use runs::store_run_summary;
#[cfg(test)]
pub(crate) use transcript::insert_message;
