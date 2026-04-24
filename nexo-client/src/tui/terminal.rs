use std::io::{self, Write, stdout};
use std::panic;
use std::process::{Command, Stdio};

use crossterm::ExecutableCommand;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

pub struct TerminalHandle {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalHandle {
    pub fn new() -> cli_helpers::Result<Self> {
        enable_raw_mode()
            .map_err(|e| cli_helpers::Error::Io(format!("Failed to enable raw mode: {e}")))?;
        let mut out = stdout();
        out.execute(EnterAlternateScreen).map_err(|e| {
            cli_helpers::Error::Io(format!("Failed to enter alternate screen: {e}"))
        })?;
        out.execute(EnableMouseCapture)
            .map_err(|e| cli_helpers::Error::Io(format!("Failed to enable mouse capture: {e}")))?;

        let terminal = Terminal::new(CrosstermBackend::new(stdout()))
            .map_err(|e| cli_helpers::Error::Io(format!("Failed to create terminal: {e}")))?;

        Ok(Self { terminal })
    }

    pub fn draw(&mut self, draw_fn: impl FnOnce(&mut ratatui::Frame<'_>)) -> cli_helpers::Result {
        self.terminal
            .draw(draw_fn)
            .map(|_| ())
            .map_err(|e| cli_helpers::Error::Io(format!("Terminal draw failed: {e}")))
    }
}

impl Drop for TerminalHandle {
    fn drop(&mut self) {
        let _ = stdout().execute(DisableMouseCapture);
        let _ = stdout().execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

pub fn install_panic_hook() {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        let _ = stdout().execute(DisableMouseCapture);
        let _ = stdout().execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
        original_hook(panic_info);
    }));
}

pub fn copy_to_clipboard(text: &str) -> cli_helpers::Result {
    #[cfg(target_os = "macos")]
    {
        return run_clipboard_command("pbcopy", &[], text);
    }

    #[cfg(target_os = "linux")]
    {
        for (program, args) in [
            ("wl-copy", &[][..]),
            ("xclip", &["-selection", "clipboard"][..]),
            ("xsel", &["--clipboard", "--input"][..]),
        ] {
            match run_clipboard_command(program, args, text) {
                Ok(()) => return Ok(()),
                Err(cli_helpers::Error::Io(error))
                    if error.contains("No such file or directory") =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }

        return Err(cli_helpers::Error::Io(
            "No supported clipboard utility found (tried wl-copy, xclip, xsel)".to_string(),
        ));
    }

    #[cfg(target_os = "windows")]
    {
        return run_clipboard_command("clip", &[], text);
    }

    #[allow(unreachable_code)]
    Err(cli_helpers::Error::Io(
        "Clipboard copy is not supported on this platform".to_string(),
    ))
}

fn run_clipboard_command(program: &str, args: &[&str], text: &str) -> cli_helpers::Result {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| cli_helpers::Error::Io(format!("Failed to launch {program}: {e}")))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| cli_helpers::Error::Io(format!("Failed to write to {program}: {e}")))?;
    }

    let status = child
        .wait()
        .map_err(|e| cli_helpers::Error::Io(format!("Failed to wait for {program}: {e}")))?;

    if status.success() {
        Ok(())
    } else {
        Err(cli_helpers::Error::Io(format!(
            "{program} exited with status {status}"
        )))
    }
}
