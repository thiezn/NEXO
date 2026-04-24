pub mod schema;
pub mod start;

use crate::cli::base::Command;

pub async fn dispatch(command: Command) -> utl_helpers::Result {
    match command {
        Command::Start {
            url,
            session,
            name,
            model,
        } => {
            start::run_start(start::StartCommand {
                url_override: url,
                session_id: session,
                session_name: name,
                model_id: model,
            })
            .await
        }
        Command::Schema { section, output } => schema::run_schema(section, output.as_deref()),
    }
}
