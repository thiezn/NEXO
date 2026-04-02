pub mod chat;
pub mod connect;
pub mod helpers;
pub mod image_analyze;
pub mod schema;

use crate::cli::base::Command;

pub async fn dispatch(command: Command) -> utl_helpers::Result {
    match command {
        Command::Connect { url } => connect::run_connect(url).await,
        Command::Chat {
            url,
            session,
            name,
            model,
        } => {
            chat::run_chat(chat::ChatOptions {
                url_override: url,
                session_id: session,
                session_name: name,
                model_id: model,
            })
            .await
        }
        Command::ImageAnalyze { image_path, prompt } => {
            image_analyze::run_image_analyze(image_path, prompt).await
        }
        Command::Schema { section, output } => schema::run_schema(section, output.as_deref()),
    }
}
