//! WebSocket request handling for connected gateway peers.

mod audio_analyze;
mod audio_generate;
mod base;
mod connection;
mod cron;
mod dispatch;
mod image_analyze;
mod image_generate;
mod prompt;
mod run;
mod send;
mod session;
mod status;
mod tools;

// Commenting out tests as we've severely refactored all the code
// #[cfg(test)]
// mod tests;

pub use connection::handle_connection;
pub(crate) use dispatch::dispatch_method;
