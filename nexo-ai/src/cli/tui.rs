//! Placeholder for the future standalone inference TUI.

use std::process::ExitCode;

use cli_helpers::{CommandContext, Runnable};

#[derive(clap::Args, Debug, Clone, Copy)]
pub(crate) struct StartCommand;

impl Runnable for StartCommand {
    type Error = cli_helpers::Error;

    fn run(self, context: &mut CommandContext) -> cli_helpers::Result<ExitCode> {
        context.stdout_line("nexo-ai TUI is not implemented yet")?;
        Ok(ExitCode::SUCCESS)
    }
}
