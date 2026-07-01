use cli_helpers::markdown::{
    MarkdownLine as RenderedMarkdownLine, MarkdownLineKind, MarkdownSpan as RenderedMarkdownSpan,
    parse_markdown,
};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
};

use super::{
    command,
    model::{CompletionState, LogEntry, LogKind, Model},
};

const COPY_ALL_LABEL: &str = "[Copy All]";
const COPY_LAST_LABEL: &str = "[Copy Last]";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ActivityLayout {
    panel_area: Rect,
    button_row_area: Rect,
    copy_all_button_area: Rect,
    copy_last_button_area: Rect,
    log_area: Rect,
    total_wrapped_lines: usize,
}

#[derive(Clone, Copy)]
struct Theme {
    border: Style,
    border_emphasis: Style,
    title: Style,
    accent: Style,
    muted: Style,
    info: Style,
    success: Style,
    warning: Style,
    danger: Style,
    selection: Style,
    placeholder: Style,
}

pub fn render(model: &mut Model, frame: &mut Frame<'_>) {
    let theme = theme();
    let sections = main_sections(model, frame.area());
    let activity_lines = activity_lines(model, theme);
    let activity_layout = activity_layout(&activity_lines, sections[1]);

    render_summary(model, frame, sections[0], theme);
    render_logs(model, frame, &activity_lines, activity_layout, theme);
    if let Some(completion) = &model.completion {
        render_completion(completion, frame, sections[2], theme);
    }
    render_input(model, frame, sections[3], theme);

    if model.show_help {
        render_help_popup(frame, theme);
    }
}

fn render_summary(model: &Model, frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(area);

    let protocol = model
        .summary
        .protocol
        .map(|protocol| protocol.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let session = model
        .current_session_id
        .as_deref()
        .or(model.default_session_name.as_deref())
        .unwrap_or("none");
    let model_id = model.default_model_id.as_deref().unwrap_or("auto");
    let clients = model
        .summary
        .connected_clients
        .map(|count| count.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let nodes = model
        .summary
        .connected_nodes
        .map(|count| count.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let capabilities = if model.summary.capabilities.is_empty() {
        "none reported".to_string()
    } else {
        summarize_value(&model.summary.capabilities.join(", "), 52)
    };
    let run_status = match &model.active_stream {
        Some(stream) => match (&stream.tool_name, &stream.tool_call_id) {
            (Some(tool_name), Some(tool_call_id)) => {
                format!("{:?} via {tool_name} ({tool_call_id})", stream.status)
            }
            (Some(tool_name), None) => format!("{:?} via {tool_name}", stream.status),
            _ => format!("{:?}", stream.status),
        },
        None => "idle".to_string(),
    };
    let error_text = model.summary.last_error.as_deref().unwrap_or("none");

    let left = vec![
        key_value_line(theme, "Gateway", &model.summary.gateway_url, theme.info),
        key_value_line(theme, "Protocol", &protocol, theme.accent),
        key_value_line(
            theme,
            "Connection",
            connection_label(model),
            connection_style(theme, model),
        ),
        key_value_line(theme, "Session", session, theme.accent),
        key_value_line(theme, "Model", model_id, theme.info),
    ];
    let right = vec![
        key_value_line(theme, "Clients", &clients, theme.success),
        key_value_line(theme, "Nodes", &nodes, theme.info),
        key_value_line(theme, "Capabilities", &capabilities, theme.accent),
        key_value_line(theme, "Run", &run_status, stream_style(theme, model)),
        key_value_line(theme, "Last Error", error_text, error_style(theme, model)),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(left))
            .block(panel_block("Summary", theme.border, theme.title))
            .wrap(Wrap { trim: false }),
        columns[0],
    );
    frame.render_widget(
        Paragraph::new(Text::from(right))
            .block(panel_block("Status", theme.border_emphasis, theme.title))
            .wrap(Wrap { trim: false }),
        columns[1],
    );
}

fn render_logs(
    model: &mut Model,
    frame: &mut Frame<'_>,
    lines: &[Line<'static>],
    layout: ActivityLayout,
    theme: Theme,
) {
    model.update_activity_view(
        layout.panel_area,
        layout.log_area.height as usize,
        layout.total_wrapped_lines,
        layout.copy_all_button_area,
        layout.copy_last_button_area,
    );

    frame.render_widget(
        panel_block("Activity", theme.border_emphasis, theme.title),
        layout.panel_area,
    );

    if layout.button_row_area.height > 0 {
        frame.render_widget(
            Paragraph::new(activity_button_bar(theme)).wrap(Wrap { trim: false }),
            layout.button_row_area,
        );
    }

    if layout.log_area.width == 0 || layout.log_area.height == 0 {
        return;
    }

    let max_scroll_offset = layout
        .total_wrapped_lines
        .saturating_sub(layout.log_area.height as usize)
        .min(u16::MAX as usize) as u16;
    let scroll_top = max_scroll_offset.saturating_sub(model.activity_scroll.min(max_scroll_offset));

    frame.render_widget(
        Paragraph::new(Text::from(lines.to_vec()))
            .scroll((scroll_top, 0))
            .wrap(Wrap { trim: false }),
        layout.log_area,
    );
}

fn render_completion(
    completion: &CompletionState,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
) {
    let items = completion
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let selected = index == completion.selected;
            let detail = if item.replacement != item.label {
                format!("  {}", item.replacement)
            } else {
                String::new()
            };
            let line = Line::from(vec![
                Span::styled(
                    if selected { "> " } else { "  " },
                    if selected {
                        theme.selection
                    } else {
                        theme.muted
                    },
                ),
                Span::styled(
                    item.label.clone(),
                    if selected {
                        theme.selection.add_modifier(Modifier::BOLD)
                    } else {
                        theme.accent
                    },
                ),
                Span::styled(detail, theme.muted),
            ]);
            let item = ListItem::new(line);
            if selected {
                item.style(theme.selection)
            } else {
                item
            }
        })
        .collect::<Vec<_>>();

    frame.render_widget(
        List::new(items).block(panel_block(
            "Autocomplete",
            theme.border_emphasis,
            theme.title,
        )),
        area,
    );
}

fn render_input(model: &Model, frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    let title = if model.input.is_empty() {
        "Command (/help or F1 for help)"
    } else {
        "Command"
    };
    let content = if model.input.is_empty() {
        Line::from(vec![
            Span::styled("> ", theme.accent),
            Span::styled(
                "Type a prompt, or use /status, /session list, or /help. Use Tab and @file completion.",
                theme.placeholder,
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("> ", theme.accent),
            Span::raw(model.input.clone()),
        ])
    };

    frame.render_widget(
        Paragraph::new(content).block(panel_block(title, theme.border_emphasis, theme.title)),
        area,
    );

    let cursor_x = area
        .x
        .saturating_add(3)
        .saturating_add(model.input[..model.cursor].chars().count() as u16)
        .min(area.x + area.width.saturating_sub(2));
    let cursor_y = area.y.saturating_add(1);
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn render_help_popup(frame: &mut Frame<'_>, theme: Theme) {
    let area = centered_rect(82, 72, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(command::help_text())
            .block(panel_block("Help", theme.border_emphasis, theme.title))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn activity_lines(model: &Model, theme: Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in &model.logs {
        append_log_entry(&mut lines, entry, theme);
    }

    if let Some(stream) = &model.active_stream {
        let title = match &stream.tool_name {
            Some(tool_name) => format!("assistant ({:?}, tool: {tool_name})", stream.status),
            None => format!("assistant ({:?})", stream.status),
        };
        append_log_entry(
            &mut lines,
            &LogEntry {
                kind: if stream.error.is_some() {
                    LogKind::Error
                } else {
                    LogKind::Response
                },
                title,
                body: if stream.content.is_empty() {
                    stream
                        .error
                        .clone()
                        .unwrap_or_else(|| "...waiting for response...".to_string())
                } else {
                    stream.content.clone()
                },
            },
            theme,
        );
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "No activity yet. Run /status, /run, or /help to get started.",
            theme.placeholder,
        )));
    }

    lines
}

fn append_log_entry(lines: &mut Vec<Line<'static>>, entry: &LogEntry, theme: Theme) {
    if uses_markdown_renderer(entry) {
        append_markdown_entry(lines, entry, theme);
        return;
    }

    let label_style = log_style(theme, entry.kind);
    let mut body_lines = entry.body.lines();
    let first_line = body_lines.next().unwrap_or_default().to_string();

    lines.push(Line::from(vec![
        Span::styled(format!("[{}] ", entry.title), label_style),
        Span::raw(first_line),
    ]));

    for line in body_lines {
        lines.push(Line::from(Span::raw(format!("  {line}"))));
    }

    lines.push(Line::raw(String::new()));
}

fn append_markdown_entry(lines: &mut Vec<Line<'static>>, entry: &LogEntry, theme: Theme) {
    let rendered = parse_markdown(&entry.body);
    let label_style = log_style(theme, entry.kind);

    for (index, line) in rendered.lines.iter().enumerate() {
        if line.is_blank() {
            lines.push(Line::raw(String::new()));
            continue;
        }

        let mut spans = Vec::new();
        if index == 0 {
            spans.push(Span::styled(format!("[{}] ", entry.title), label_style));
        } else {
            spans.push(Span::raw("  ".to_string()));
        }
        spans.extend(render_markdown_line(line, theme));
        lines.push(Line::from(spans));
    }

    if rendered.lines.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("[{}] ", entry.title),
            label_style,
        )));
    }

    lines.push(Line::raw(String::new()));
}

fn render_markdown_line(line: &RenderedMarkdownLine, theme: Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    if !line.prefix.is_empty() {
        spans.push(Span::styled(
            line.prefix.clone(),
            markdown_prefix_style(theme, &line.kind),
        ));
    }

    for span in &line.spans {
        spans.push(Span::styled(
            span.text.clone(),
            markdown_span_style(theme, &line.kind, span),
        ));
    }

    spans
}

fn uses_markdown_renderer(entry: &LogEntry) -> bool {
    matches!(entry.kind, LogKind::Response) || entry.title.starts_with("assistant")
}

fn markdown_prefix_style(theme: Theme, kind: &MarkdownLineKind) -> Style {
    match kind {
        MarkdownLineKind::Heading { .. } => theme.accent.add_modifier(Modifier::BOLD),
        MarkdownLineKind::Quote => theme.muted.add_modifier(Modifier::ITALIC),
        MarkdownLineKind::ListItem => theme.accent.add_modifier(Modifier::BOLD),
        MarkdownLineKind::CodeBlock { .. } => theme.muted.bg(Color::Rgb(24, 34, 48)),
        MarkdownLineKind::Paragraph | MarkdownLineKind::Blank => Style::default(),
    }
}

fn markdown_span_style(
    theme: Theme,
    kind: &MarkdownLineKind,
    span: &RenderedMarkdownSpan,
) -> Style {
    let mut style = match kind {
        MarkdownLineKind::Heading { level } => heading_style(theme, *level),
        MarkdownLineKind::Quote => theme.muted.add_modifier(Modifier::ITALIC),
        MarkdownLineKind::ListItem => Style::default(),
        MarkdownLineKind::CodeBlock { .. } => Style::default()
            .fg(Color::Rgb(230, 236, 244))
            .bg(Color::Rgb(24, 34, 48)),
        MarkdownLineKind::Paragraph | MarkdownLineKind::Blank => Style::default(),
    };

    if span.style.strong {
        style = style.add_modifier(Modifier::BOLD);
    }
    if span.style.emphasis {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if span.style.strikethrough {
        style = style.add_modifier(Modifier::CROSSED_OUT);
    }
    if span.style.link {
        style = style.fg(theme.info.fg.unwrap_or(Color::Cyan));
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if span.style.code && !matches!(kind, MarkdownLineKind::CodeBlock { .. }) {
        style = style
            .fg(Color::Rgb(230, 236, 244))
            .bg(Color::Rgb(32, 45, 63));
    }

    style
}

fn heading_style(theme: Theme, level: u8) -> Style {
    match level {
        1 => theme
            .title
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        2 => theme.title.add_modifier(Modifier::BOLD),
        3 => theme.accent.add_modifier(Modifier::BOLD),
        _ => theme.accent,
    }
}

fn theme() -> Theme {
    Theme {
        border: Style::default().fg(Color::Rgb(102, 120, 138)),
        border_emphasis: Style::default().fg(Color::Rgb(120, 164, 255)),
        title: Style::default()
            .fg(Color::Rgb(232, 237, 243))
            .add_modifier(Modifier::BOLD),
        accent: Style::default().fg(Color::Rgb(126, 214, 223)),
        muted: Style::default().fg(Color::Rgb(142, 154, 175)),
        info: Style::default().fg(Color::Rgb(105, 184, 255)),
        success: Style::default().fg(Color::Rgb(104, 211, 145)),
        warning: Style::default().fg(Color::Rgb(245, 189, 92)),
        danger: Style::default().fg(Color::Rgb(255, 120, 117)),
        selection: Style::default()
            .fg(Color::Rgb(15, 23, 42))
            .bg(Color::Rgb(126, 214, 223)),
        placeholder: Style::default()
            .fg(Color::Rgb(126, 138, 158))
            .add_modifier(Modifier::ITALIC),
    }
}

fn panel_block<'a>(title: &'a str, border_style: Style, title_style: Style) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(Line::from(Span::styled(title, title_style)))
}

fn key_value_line(theme: Theme, label: &str, value: &str, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<14}"), theme.muted),
        Span::styled(value.to_string(), value_style),
    ])
}

fn connection_style(theme: Theme, model: &Model) -> Style {
    if !model.summary.connected {
        theme.danger
    } else if model.summary.last_error.is_some() {
        theme.warning
    } else {
        theme.success
    }
}

fn connection_label(model: &Model) -> &str {
    if model.summary.connected {
        "connected"
    } else {
        "disconnected"
    }
}

fn stream_style(theme: Theme, model: &Model) -> Style {
    if let Some(stream) = &model.active_stream {
        if stream.error.is_some() {
            theme.warning
        } else {
            theme.accent
        }
    } else {
        theme.muted
    }
}

fn error_style(theme: Theme, model: &Model) -> Style {
    if model.summary.last_error.is_some() {
        theme.danger
    } else {
        theme.muted
    }
}

fn log_style(theme: Theme, kind: LogKind) -> Style {
    match kind {
        LogKind::Info => theme.info,
        LogKind::Success => theme.success,
        LogKind::Warning => theme.warning,
        LogKind::Error => theme.danger.add_modifier(Modifier::BOLD),
        LogKind::Command => theme.accent.add_modifier(Modifier::BOLD),
        LogKind::Event => theme.info,
        LogKind::Response => theme.title,
    }
}

fn main_sections(model: &Model, area: Rect) -> [Rect; 4] {
    let completion_height = model
        .completion
        .as_ref()
        .map(|completion| (completion.items.len() as u16).min(6) + 2)
        .unwrap_or(0);

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Min(8),
            Constraint::Length(completion_height),
            Constraint::Length(3),
        ])
        .split(area);

    [sections[0], sections[1], sections[2], sections[3]]
}

fn activity_layout(lines: &[Line<'static>], panel_area: Rect) -> ActivityLayout {
    let inner = block_inner(panel_area);
    if inner.width == 0 || inner.height == 0 {
        return ActivityLayout {
            panel_area,
            ..ActivityLayout::default()
        };
    }

    let button_row_area = Rect::new(inner.x, inner.y, inner.width, 1);
    let log_area = Rect::new(
        inner.x,
        inner.y.saturating_add(1),
        inner.width,
        inner.height.saturating_sub(1),
    );
    let copy_all_button_area = Rect::new(
        button_row_area.x,
        button_row_area.y,
        COPY_ALL_LABEL.len() as u16,
        1,
    );
    let copy_last_button_area = Rect::new(
        copy_all_button_area
            .x
            .saturating_add(copy_all_button_area.width + 1),
        button_row_area.y,
        COPY_LAST_LABEL.len() as u16,
        1,
    );
    let total_wrapped_lines = if log_area.width == 0 {
        0
    } else {
        lines
            .iter()
            .map(|line| wrapped_line_height(line, log_area.width) as usize)
            .sum()
    };

    ActivityLayout {
        panel_area,
        button_row_area,
        copy_all_button_area,
        copy_last_button_area,
        log_area,
        total_wrapped_lines,
    }
}

fn block_inner(area: Rect) -> Rect {
    Rect::new(
        area.x.saturating_add(1),
        area.y.saturating_add(1),
        area.width.saturating_sub(2),
        area.height.saturating_sub(2),
    )
}

fn activity_button_bar(theme: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(COPY_ALL_LABEL, theme.info.add_modifier(Modifier::BOLD)),
        Span::styled(" ", theme.muted),
        Span::styled(COPY_LAST_LABEL, theme.accent.add_modifier(Modifier::BOLD)),
        Span::styled("  mouse wheel scrolls activity", theme.muted),
    ])
}

fn wrapped_line_height(line: &Line<'_>, width: u16) -> u16 {
    if width == 0 {
        return 0;
    }

    let line_width = line.width() as u16;
    line_width.max(1).div_ceil(width)
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn summarize_value(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let truncated: String = value.chars().take(max_chars.saturating_sub(3)).collect();
    format!("{truncated}...")
}
