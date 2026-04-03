use std::error::Error as StdError;
use std::fmt;

pub type Result<T = ()> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum Error {
    Config(String),
    Io(String),
    Network(String),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(msg) => write!(f, "Config error: {msg}"),
            Error::Io(msg) => write!(f, "IO error: {msg}"),
            Error::Network(msg) => write!(f, "Network error: {msg}"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl StdError for Error {}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Error::Other(format!("Arc Lock poisoned: {}", e))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

#[cfg(feature = "output")]
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Other(e.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_owned())
    }
}
