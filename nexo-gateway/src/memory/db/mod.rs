/// The base module for the database
pub mod base;
pub use base::DbClient;

/// Implementation of DbClient related to queue persistence.
pub mod agent_job_queue;
/// Internal typed records and serde helpers for DB persistence.
mod db_types;
/// Implementation of DbClient related to inference intent persistence.
pub mod inference_intents;
/// Implementation of DbClient related to inference run persistence.
pub mod inference_runs;
/// Implementation of DbClient related to operation ownership persistence.
pub mod operations;
/// Implementation of DbClient related to peer persistence.
pub mod peers;
/// Implementation of DbClient related to tool-definition persistence.
pub mod tools;
