//! Private any-tts runtime integration for local speech generation.

mod runtime;

pub use runtime::AnyTtsModelConfig;
pub(crate) use runtime::{AnyTtsRuntime, internal_runtime_kind};
