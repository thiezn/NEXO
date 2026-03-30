pub mod init;
pub mod models_list;
pub mod models_pull;
pub mod start;

use crate::cli::base::{Command, ModelsCommand};

pub async fn dispatch(command: Command) -> utl_helpers::Result {
    match command {
        Command::Init => init::run(),
        Command::Start { url } => start::run(url).await,
        Command::Models { action } => match action {
            ModelsCommand::Pull { model, force } => models_pull::run(&model, force).await,
            ModelsCommand::List => models_list::run(),
        },
    }
}
