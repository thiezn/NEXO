use crate::tui;

#[derive(Debug, Clone, Default)]
pub struct StartCommand {
    pub url_override: Option<String>,
    pub session_id: Option<String>,
    pub session_name: Option<String>,
    pub model_id: Option<String>,
}

pub async fn run_start(command: StartCommand) -> utl_helpers::Result {
    tui::run_start(tui::StartOptions {
        url_override: command.url_override,
        initial_session_id: command.session_id,
        initial_session_name: command.session_name,
        initial_model_id: command.model_id,
    })
    .await
}
