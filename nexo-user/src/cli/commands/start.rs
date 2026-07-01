use super::{save_user_properties, user_config_path};
use nexo_core::{ClientInfo, DeviceInfo, UserProperties};
use nexo_user::{NexoUser, Result};
use tokio::time::{Duration, sleep};
use tracing::{info, warn};

#[derive(Debug, Clone, Default)]
pub struct StartCommand {
    /// Optional gateway URL override for this launch.
    pub url: Option<String>,
}

/// Run the `start` command, which launches the interactive NEXO terminal UI.
pub async fn run(command: StartCommand) -> Result {
    let path = user_config_path();
    let mut properties = if path.exists() {
        let properties: UserProperties = cli_helpers::config::load(&path)?;
        properties.into_builder().build()
    } else {
        let properties = UserProperties::new(
            ClientInfo::new(env!("CARGO_PKG_VERSION")),
            DeviceInfo::default(),
            nexo_ws_schema::AUTH_TOKEN,
        );
        save_user_properties(&properties)?;
        properties
    };
    if let Some(url) = command.url {
        properties = properties.into_builder().gateway_url(url).build();
    }

    info!(
        "Starting nexo-user '{}' v{}",
        properties.client().id,
        properties.client().version
    );

    let engine = NexoUser::new(properties);

    loop {
        match engine.run().await {
            Ok(()) => return Ok(()),
            Err(_) => {
                warn!("retrying in 10 seconds...");
                sleep(Duration::from_secs(10)).await;
            }
        }
    }
}
