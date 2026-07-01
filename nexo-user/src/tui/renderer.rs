use super::state::{HistoryEntry, HistoryKind, HistorySource};
use super::{InputState, NexoUserState};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

const NORD_POLAR_NIGHT_0: Color = Color::Rgb(46, 52, 64);
const NORD_POLAR_NIGHT_1: Color = Color::Rgb(59, 66, 82);
const NORD_POLAR_NIGHT_3: Color = Color::Rgb(76, 86, 106);
const NORD_SNOW_STORM_0: Color = Color::Rgb(216, 222, 233);
const NORD_SNOW_STORM_1: Color = Color::Rgb(229, 233, 240);
const NORD_FROST_1: Color = Color::Rgb(136, 192, 208);
const NORD_FROST_2: Color = Color::Rgb(129, 161, 193);
const NORD_FROST_3: Color = Color::Rgb(94, 129, 172);
const NORD_AURORA_RED: Color = Color::Rgb(191, 97, 106);
const NORD_AURORA_YELLOW: Color = Color::Rgb(235, 203, 139);
const NORD_AURORA_GREEN: Color = Color::Rgb(163, 190, 140);
const NORD_AURORA_PURPLE: Color = Color::Rgb(180, 142, 173);

/// Renders the full three-pane TUI.
///
/// # Arguments
///
/// * `frame` - The ratatui frame being drawn.
/// * `state` - The current engine-owned application state.
/// * `input_state` - The local prompt and viewport state.
pub fn render(frame: &mut Frame<'_>, state: &NexoUserState, input_state: &InputState) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(5),
        ])
        .split(frame.area());

    render_summary(frame, sections[0], state);
    render_history(frame, sections[1], state, input_state);
    render_prompt(frame, sections[2], state, input_state);
}

/// Renders the top summary section.
///
/// # Arguments
///
/// * `frame` - The ratatui frame being drawn.
/// * `area` - The screen area allocated to the summary pane.
/// * `state` - The current engine-owned application state.
fn render_summary(frame: &mut Frame<'_>, area: Rect, state: &NexoUserState) {
    let selected_session = state
        .selected_session_id()
        .map(ToString::to_string)
        .unwrap_or_else(|| "none".into());

    let lines = vec![
        Line::from(vec![
            Span::styled("Connection: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{:?}", state.connection_status()),
                Style::default().fg(NORD_FROST_1),
            ),
            Span::raw("    "),
            Span::styled("Sessions: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(state.sessions().len().to_string(), Style::default().fg(NORD_SNOW_STORM_1)),
            Span::raw("    "),
            Span::styled("Active Ops: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                state.active_operations().len().to_string(),
                Style::default().fg(NORD_AURORA_YELLOW),
            ),
        ]),
        Line::from(vec![
            Span::styled("Selected Session: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(selected_session, Style::default().fg(NORD_SNOW_STORM_1)),
        ]),
        Line::from(vec![
            Span::styled("Gateway State: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                if state.state().is_some() {
                    "available"
                } else {
                    "not loaded"
                },
                Style::default().fg(NORD_SNOW_STORM_1),
            ),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(Style::default().fg(NORD_SNOW_STORM_0).bg(NORD_POLAR_NIGHT_0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(NORD_POLAR_NIGHT_3))
                    .title(Span::styled("Nexo User", Style::default().fg(NORD_FROST_2))),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Renders the middle history pane.
///
/// # Arguments
///
/// * `frame` - The ratatui frame being drawn.
/// * `area` - The screen area allocated to the history pane.
/// * `state` - The current engine-owned application state.
/// * `input_state` - The local viewport state containing the current scroll offset.
fn render_history(frame: &mut Frame<'_>, area: Rect, state: &NexoUserState, input_state: &InputState) {
    let mut lines = state
        .timeline()
        .iter()
        .flat_map(history_entry_lines)
        .collect::<Vec<_>>();

    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "No activity yet. Use /state or type a prompt to get started.",
            Style::default().fg(NORD_POLAR_NIGHT_3),
        )]));
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(Style::default().fg(NORD_SNOW_STORM_0).bg(NORD_POLAR_NIGHT_1))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(NORD_POLAR_NIGHT_3))
                    .title(Span::styled("History", Style::default().fg(NORD_FROST_2))),
            )
            .scroll((input_state.history_scroll() as u16, 0))
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Converts one structured history entry into rendered lines.
///
/// # Arguments
///
/// * `entry` - The structured history entry to render.
fn history_entry_lines(entry: &HistoryEntry) -> Vec<Line<'static>> {
    let (source_label, source_style) = match entry.source {
        HistorySource::User => ("user", Style::default().fg(NORD_FROST_1).add_modifier(Modifier::BOLD)),
        HistorySource::Gateway => (
            "gateway",
            Style::default().fg(NORD_AURORA_YELLOW).add_modifier(Modifier::BOLD),
        ),
        HistorySource::Inference => (
            "inference",
            Style::default().fg(NORD_AURORA_GREEN).add_modifier(Modifier::BOLD),
        ),
        HistorySource::Tool => ("tool", Style::default().fg(NORD_AURORA_PURPLE).add_modifier(Modifier::BOLD)),
        HistorySource::Error => ("error", Style::default().fg(NORD_AURORA_RED).add_modifier(Modifier::BOLD)),
    };

    let body_style = match entry.kind {
        HistoryKind::UserPrompt => Style::default().fg(NORD_SNOW_STORM_1),
        HistoryKind::UserCommand => Style::default()
            .fg(NORD_SNOW_STORM_1)
            .add_modifier(Modifier::ITALIC),
        HistoryKind::Error => Style::default()
            .fg(NORD_SNOW_STORM_1)
            .add_modifier(Modifier::BOLD),
        HistoryKind::InferenceText => Style::default().fg(NORD_SNOW_STORM_0),
        HistoryKind::GatewayControl
        | HistoryKind::GatewayState
        | HistoryKind::Operation
        | HistoryKind::InferenceThinking
        | HistoryKind::ToolActivity => Style::default().fg(NORD_POLAR_NIGHT_3),
    };

    let body_lines = if entry.body.is_empty() {
        vec![""]
    } else {
        entry.body.split('\n').collect::<Vec<_>>()
    };

    body_lines
        .into_iter()
        .map(|body_line| {
            Line::from(vec![
                Span::styled(format!("{source_label}: "), source_style),
                Span::styled(body_line.to_owned(), body_style),
            ])
        })
        .collect()
}

/// Renders the bottom prompt and autocomplete section.
///
/// # Arguments
///
/// * `frame` - The ratatui frame being drawn.
/// * `area` - The screen area allocated to the prompt pane.
/// * `state` - The current engine-owned application state.
/// * `input_state` - The local prompt state to render.
fn render_prompt(frame: &mut Frame<'_>, area: Rect, _state: &NexoUserState, input_state: &InputState) {
    let title = if input_state.buffer().starts_with('/') {
        "Command"
    } else {
        "Prompt"
    };

    let help_line = if input_state.buffer().starts_with('/') {
        "Slash command mode. Tab completes the current command suggestion."
    } else {
        "Plain text submits a multimodal text inference request."
    };

    let content = vec![
        Line::from(vec![
            Span::styled("> ", Style::default().fg(NORD_FROST_1)),
            Span::styled(input_state.buffer().to_owned(), Style::default().fg(NORD_SNOW_STORM_1)),
        ]),
        Line::from(vec![Span::styled(help_line, Style::default().fg(NORD_POLAR_NIGHT_3))]),
        autocomplete_line(input_state),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(content))
            .style(Style::default().fg(NORD_SNOW_STORM_0).bg(NORD_POLAR_NIGHT_0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(NORD_POLAR_NIGHT_3))
                    .title(Span::styled(title, Style::default().fg(NORD_FROST_3))),
            )
            .wrap(Wrap { trim: false }),
        area,
    );

    let cursor_x = area
        .x
        .saturating_add(2)
        .saturating_add(input_state.cursor() as u16)
        .min(area.x + area.width.saturating_sub(2));
    let cursor_y = area.y.saturating_add(1);
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// Renders the current autocomplete suggestion line.
///
/// # Arguments
///
/// * `input_state` - The current prompt state containing autocomplete suggestions.
fn autocomplete_line(input_state: &InputState) -> Line<'static> {
    match input_state
        .autocomplete_selected()
        .and_then(|index| input_state.autocomplete_items().get(index))
    {
        Some(suggestion) => Line::from(vec![
            Span::styled(
                "suggestion: ",
                Style::default()
                    .fg(NORD_FROST_2)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(suggestion.clone(), Style::default().fg(NORD_SNOW_STORM_0)),
        ]),
        None => Line::from(vec![Span::raw("")]),
    }
}
