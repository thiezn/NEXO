mod command;
mod completion;
mod message;
mod model;
mod network;
mod terminal;
mod update;
mod view;

pub use model::StartOptions;

use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyEventKind};
use message::Message;
use model::{Model, RunningState};
use network::{NetworkCommand, NetworkEvent};
use update::Effect;

pub async fn run_start(options: StartOptions) -> utl_helpers::Result {
    let workspace_root = std::env::current_dir().map_err(|e| {
        utl_helpers::Error::Io(format!("Failed to determine working directory: {e}"))
    })?;

    let (connection, network_tx, mut network_rx) =
        network::connect(options.url_override.as_deref()).await?;

    terminal::install_panic_hook();
    let mut terminal = terminal::TerminalHandle::new()?;
    let mut model = Model::new(connection, options, workspace_root);

    process_message(&mut model, Message::Tick, &network_tx);

    let mut last_tick = Instant::now();
    while model.running_state == RunningState::Running {
        drain_network(&mut model, &mut network_rx, &network_tx);

        terminal.draw(|frame| view::render(&mut model, frame))?;

        if event::poll(Duration::from_millis(50))
            .map_err(|e| utl_helpers::Error::Io(format!("Terminal event poll failed: {e}")))?
        {
            let event = event::read()
                .map_err(|e| utl_helpers::Error::Io(format!("Terminal event read failed: {e}")))?;
            match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if let Some(message) = message::from_key_event(key) {
                        process_message(&mut model, message, &network_tx);
                    }
                }
                Event::Mouse(mouse) => {
                    if let Some(message) = message::from_mouse_event(mouse) {
                        process_message(&mut model, message, &network_tx);
                    }
                }
                Event::Resize(_, _) => process_message(&mut model, Message::Tick, &network_tx),
                _ => {}
            }
        }

        if last_tick.elapsed() >= Duration::from_millis(250) {
            process_message(&mut model, Message::Tick, &network_tx);
            last_tick = Instant::now();
        }
    }

    let _ = network_tx.send(NetworkCommand::Close);
    Ok(())
}

fn drain_network(
    model: &mut Model,
    network_rx: &mut tokio::sync::mpsc::UnboundedReceiver<NetworkEvent>,
    network_tx: &tokio::sync::mpsc::UnboundedSender<NetworkCommand>,
) {
    while let Ok(event) = network_rx.try_recv() {
        process_message(model, Message::Network(event), network_tx);
    }
}

fn process_message(
    model: &mut Model,
    message: Message,
    network_tx: &tokio::sync::mpsc::UnboundedSender<NetworkCommand>,
) {
    let effects = update::update(model, message);
    handle_effects(model, effects, network_tx);
}

fn handle_effects(
    model: &mut Model,
    effects: Vec<Effect>,
    network_tx: &tokio::sync::mpsc::UnboundedSender<NetworkCommand>,
) {
    for effect in effects {
        match effect {
            Effect::Send(frame) => {
                if network_tx.send(NetworkCommand::Send(frame)).is_err() {
                    model.set_disconnected("Network writer is no longer available");
                }
            }
            Effect::CopyToClipboard { label, text } => match terminal::copy_to_clipboard(&text) {
                Ok(()) => model.push_log(
                    model::LogKind::Success,
                    "clipboard",
                    format!("Copied {label} to clipboard"),
                ),
                Err(error) => model.push_log(
                    model::LogKind::Error,
                    "clipboard",
                    format!("Failed to copy {label}: {error}"),
                ),
            },
            Effect::Close => {
                let _ = network_tx.send(NetworkCommand::Close);
            }
        }
    }
}
