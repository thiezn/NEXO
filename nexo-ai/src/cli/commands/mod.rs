pub mod pull;
pub mod list;
pub mod start;

use crate::cli::base::Command;

pub async fn dispatch(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Pull { model, force } => pull::run(&model, force).await,
        Command::List => list::run(),
        Command::Start { categories } => start::run(categories).await,
    }
}
