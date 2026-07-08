//! Persistent storage helpers used by the gateway.

/// SQLite persistence helpers for users, devices, and gateway state.
pub mod db;
/// Git-backed storage used for notes and prefill content.
pub mod git;

/// Implementation of the NotesStorage, on top of our GitStorage.
pub mod notes;
