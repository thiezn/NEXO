use crate::{NexoAi, NexoAiConfig};

/// Builder for creating a `nexo-ai` service from declarative configuration.
#[derive(Debug, Clone)]
pub struct NexoAiBuilder {
    config: NexoAiConfig,
}

impl NexoAiBuilder {
    /// Creates a new builder from the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The declarative runtime configuration.
    pub fn new(config: NexoAiConfig) -> Self {
        Self { config }
    }

    /// Returns the current builder configuration.
    pub fn config(&self) -> &NexoAiConfig {
        &self.config
    }

    /// Builds the service instance.
    pub async fn build(self) -> crate::Result<NexoAi> {
        NexoAi::from_config(self.config).await
    }
}
