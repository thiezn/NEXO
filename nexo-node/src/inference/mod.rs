//! Inference execution, model lifecycle, and KV-cache state management.

mod cache;
mod dispatch;
mod model_lifecycle;

pub(crate) use cache::SessionCacheManager;
pub(crate) use dispatch::{dispatch_image_analyze, dispatch_run_round};
pub(crate) use model_lifecycle::{handle_model_load, handle_model_unload};
