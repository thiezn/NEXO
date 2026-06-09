//! Persistent storage helpers used by the gateway.

/// Git-backed storage used for notes and prefill content.
pub mod git;
/// SQLite persistence helpers for users, devices, and gateway state.
pub mod persistent;

/// Implementation of the NotesStorage, on top of our GitStorage.
pub mod notes;
