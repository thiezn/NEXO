use crate::tui;
use nexo_core::{ClientInfo, DeviceInfo, UserProperties};

use super::{save_user_properties, user_config_path};

#[derive(Debug, Clone, Default)]
pub struct StartCommand {
    pub url_override: Option<String>,
    pub session_id: Option<String>,
    pub session_name: Option<String>,
    pub model_id: Option<String>,
}

pub async fn run_start(command: StartCommand) -> cli_helpers::Result {
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
    if let Some(url) = command.url_override {
        properties = properties.into_builder().gateway_url(url).build();
    }

    tui::run_start(tui::StartOptions {
        user: properties,
        initial_session_id: command.session_id,
        initial_session_name: command.session_name,
        initial_model_id: command.model_id,
    })
    .await
}
