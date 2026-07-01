//! mold-backed runtime integration.

mod config;
mod runtime;

pub(crate) use config::{MoldModelConfig, MoldRuntimeConfig};
pub(crate) use runtime::MoldRuntime;
