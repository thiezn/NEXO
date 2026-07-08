/// The base module for the database
pub mod base;
pub use base::DbClient;

/// Implementation of DbClient related to conversations
pub mod conversations;
