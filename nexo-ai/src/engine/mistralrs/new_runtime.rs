/// NexoAI MistralRs runtime wrapper
pub(crate) struct MistralRsRuntime {
    /// The loaded MistralRs runtime, if it has been initialized.
    ///
    /// NOTE: The MistralRs does not seem to support loading it
    /// without actually loading a model in memory.
    runtime: Option<Arc<mistralrs_core::MistralRs>>,
}

impl MistralRsRuntime {
    pub fn new() -> Self {
        Self { runtime: None }
    }
}
