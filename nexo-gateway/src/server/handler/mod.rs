//! WebSocket request handling for connected gateway peers.

mod run;
mod base;
mod connection;
mod cron;
mod dispatch;
mod image_analyze;
mod prompt;
mod send;
mod status;
mod tools;

#[cfg(test)]
mod tests;

pub use connection::handle_connection;
pub(crate) use dispatch::dispatch_method;
