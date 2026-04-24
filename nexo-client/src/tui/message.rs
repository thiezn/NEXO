use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use super::network::NetworkEvent;

#[derive(Debug)]
pub enum Message {
    Tick,
    Network(NetworkEvent),
    SubmitInput,
    ShowHelp(bool),
    InsertChar(char),
    Backspace,
    Delete,
    MoveCursorLeft,
    MoveCursorRight,
    MoveCursorHome,
    MoveCursorEnd,
    AcceptCompletion,
    SelectNextCompletion,
    SelectPrevCompletion,
    ClearCompletion,
    ScrollActivityUp { column: u16, row: u16 },
    ScrollActivityDown { column: u16, row: u16 },
    Click { column: u16, row: u16 },
    Quit,
}

pub fn from_key_event(key: KeyEvent) -> Option<Message> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            Some(Message::Quit)
        }
        (KeyCode::Enter, _) => Some(Message::SubmitInput),
        (KeyCode::F(1), _) => Some(Message::ShowHelp(true)),
        (KeyCode::Backspace, _) => Some(Message::Backspace),
        (KeyCode::Delete, _) => Some(Message::Delete),
        (KeyCode::Left, _) => Some(Message::MoveCursorLeft),
        (KeyCode::Right, _) => Some(Message::MoveCursorRight),
        (KeyCode::Home, _) => Some(Message::MoveCursorHome),
        (KeyCode::End, _) => Some(Message::MoveCursorEnd),
        (KeyCode::Tab, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
            Some(Message::SelectPrevCompletion)
        }
        (KeyCode::BackTab, _) => Some(Message::SelectPrevCompletion),
        (KeyCode::Tab, _) => Some(Message::AcceptCompletion),
        (KeyCode::Up, _) => Some(Message::SelectPrevCompletion),
        (KeyCode::Down, _) => Some(Message::SelectNextCompletion),
        (KeyCode::Esc, _) => Some(Message::ClearCompletion),
        (KeyCode::Char(ch), modifiers)
            if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            Some(Message::InsertChar(ch))
        }
        _ => None,
    }
}

pub fn from_mouse_event(mouse: MouseEvent) -> Option<Message> {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => Some(Message::Click {
            column: mouse.column,
            row: mouse.row,
        }),
        MouseEventKind::ScrollUp => Some(Message::ScrollActivityUp {
            column: mouse.column,
            row: mouse.row,
        }),
        MouseEventKind::ScrollDown => Some(Message::ScrollActivityDown {
            column: mouse.column,
            row: mouse.row,
        }),
        _ => None,
    }
}
