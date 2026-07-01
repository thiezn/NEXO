use std::fs;
use std::path::Path;

use super::command;
use super::model::{CompletionItem, CompletionState};

const MAX_RESULTS: usize = 8;
const MAX_DEPTH: usize = 6;
const SKIP_DIRS: &[&str] = &[".git", "target", "datasets", "logs", "coverage", "tmp"];

pub fn compute(workspace_root: &Path, input: &str, cursor: usize) -> Option<CompletionState> {
    if input.is_empty() || cursor > input.len() {
        return None;
    }

    if let Some(completion) = command_completion_items(input, cursor) {
        return Some(completion);
    }

    let (start, end) = token_range(input, cursor);
    let token = &input[start..end];

    if token.starts_with('@') {
        let items = file_completion_items(workspace_root, token);
        return (!items.is_empty()).then_some(CompletionState {
            items,
            selected: 0,
            range: (start, end),
        });
    }

    None
}

fn command_completion_items(input: &str, cursor: usize) -> Option<CompletionState> {
    let prefix = input.get(..cursor)?.strip_prefix('/')?;
    let items = command::COMMAND_NAMES
        .iter()
        .filter(|command| command.starts_with(prefix))
        .map(|command| CompletionItem {
            label: format!("/{command}"),
            replacement: format!("/{command} "),
        })
        .take(MAX_RESULTS)
        .collect::<Vec<_>>();

    (!items.is_empty()).then_some(CompletionState {
        items,
        selected: 0,
        range: (0, cursor),
    })
}

fn file_completion_items(workspace_root: &Path, token: &str) -> Vec<CompletionItem> {
    let prefix = token.trim_start_matches('@');
    let mut candidates = if prefix.contains('/') {
        list_in_directory(workspace_root, prefix)
    } else {
        search_workspace(workspace_root, prefix)
    };
    candidates.sort();
    candidates.dedup();
    candidates
        .into_iter()
        .take(MAX_RESULTS)
        .map(|candidate| CompletionItem {
            label: format!("@{candidate}"),
            replacement: format!("@{candidate}"),
        })
        .collect()
}

fn list_in_directory(workspace_root: &Path, prefix: &str) -> Vec<String> {
    let (dir_part, name_prefix) = match prefix.rsplit_once('/') {
        Some((dir, name)) => (dir, name),
        None => ("", prefix),
    };

    let base_dir = if dir_part.is_empty() {
        workspace_root.to_path_buf()
    } else {
        workspace_root.join(dir_part)
    };

    let Ok(entries) = fs::read_dir(&base_dir) else {
        return Vec::new();
    };

    let mut matches = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with(name_prefix) {
            continue;
        }

        let mut relative = if dir_part.is_empty() {
            file_name.to_string()
        } else {
            format!("{dir_part}/{file_name}")
        };
        if entry.path().is_dir() {
            relative.push('/');
        }
        matches.push(relative);
    }
    matches
}

fn search_workspace(workspace_root: &Path, prefix: &str) -> Vec<String> {
    let mut matches = Vec::new();
    let mut stack = vec![(workspace_root.to_path_buf(), 0usize)];

    while let Some((dir, depth)) = stack.pop() {
        if depth > MAX_DEPTH || matches.len() >= MAX_RESULTS {
            continue;
        }

        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };

        for entry in entries.flatten() {
            if matches.len() >= MAX_RESULTS {
                break;
            }

            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            if path.is_dir() {
                if SKIP_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                stack.push((path, depth + 1));
                continue;
            }

            let relative = path
                .strip_prefix(workspace_root)
                .ok()
                .and_then(Path::to_str);
            if (name.starts_with(prefix)
                || relative.is_some_and(|relative| relative.starts_with(prefix)))
                && let Some(relative) = relative
            {
                matches.push(relative.replace('\\', "/"));
            }
        }
    }

    matches
}

fn token_range(input: &str, cursor: usize) -> (usize, usize) {
    let mut start = cursor;
    while start > 0 && !input.as_bytes()[start - 1].is_ascii_whitespace() {
        start -= 1;
    }

    let mut end = cursor;
    while end < input.len() && !input.as_bytes()[end].is_ascii_whitespace() {
        end += 1;
    }

    (start, end)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn command_completion_matches_slash_prefix() {
        let completion = compute(Path::new("."), "/sta", 4).unwrap();
        assert!(
            completion
                .items
                .iter()
                .any(|item| item.replacement == "/status ")
        );
    }

    #[test]
    fn command_completion_matches_spaced_subcommand_prefix() {
        let completion = compute(Path::new("."), "/session l", 10).unwrap();
        assert!(
            completion
                .items
                .iter()
                .any(|item| item.replacement == "/session list ")
        );
    }

    #[test]
    fn token_range_selects_current_word() {
        let (start, end) = token_range("/analyze image @docs/src", 20);
        assert_eq!(&"/analyze image @docs/src"[start..end], "@docs/src");
    }
}
