use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().ok_or_else(|| Error::Config("Unable to determine home directory".to_string()))
}

pub fn resolve_path(path: &Path) -> Result<PathBuf> {
    if let Some(path_str) = path.to_str()
        && let Some(rest) = path_str.strip_prefix("~/")
    {
        return Ok(home_dir()?.join(rest));
    }

    if path == Path::new("~") {
        return home_dir();
    }

    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir()
        .map_err(|e| Error::Config(format!("Unable to determine current directory: {e}")))?;
    Ok(cwd.join(path))
}

pub fn resolve_path_str(path: &str) -> Result<PathBuf> {
    resolve_path(Path::new(path))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn resolve_absolute_path() {
        let path = resolve_path(Path::new("/tmp/test.toml")).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/test.toml"));
    }

    #[test]
    fn resolve_tilde_path() {
        let path = resolve_path(Path::new("~/test.toml")).unwrap();
        let home = dirs::home_dir().unwrap();
        assert_eq!(path, home.join("test.toml"));
    }

    #[test]
    fn resolve_relative_path() {
        let path = resolve_path(Path::new("test.toml")).unwrap();
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(path, cwd.join("test.toml"));
    }
}
