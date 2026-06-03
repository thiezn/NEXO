//! WebSocket request handling for connected gateway peers.

mod base;
mod audio_analyze;
mod connection;
mod cron;
mod dispatch;
mod image_analyze;
mod prompt;
mod run;
mod send;
mod session;
mod status;
mod tools;

#[cfg(test)]
mod tests;

pub use connection::handle_connection;
pub(crate) use dispatch::dispatch_method;
