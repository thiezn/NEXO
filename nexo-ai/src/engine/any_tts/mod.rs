//! Private any-tts runtime integration for local speech generation.

mod runtime;

pub(crate) use runtime::{
    AnyTtsRuntime, INTERNAL_RUNTIME_KEY, KOKORO_RUNTIME_ID, internal_runtime_kind,
};
