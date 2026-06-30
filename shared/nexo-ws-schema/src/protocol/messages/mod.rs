/// Helper definitions for common message types like Responses and Events
pub mod base;

/// Message definitions related to connecting and disconnecting clients.
pub mod connect;

/// Message definitions related to control operations like canceling requests.
pub mod control;

/// Message definitions related to AI inference operations.
pub mod inference;

/// Message definitions related to tool operations.
pub mod tools;
