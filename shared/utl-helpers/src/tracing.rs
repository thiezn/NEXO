use clap::ValueEnum;

/// Log level enum usable as a clap `ValueEnum`.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }

    pub fn to_tracing_level(self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let v = value.trim();
        if v.eq_ignore_ascii_case("trace") {
            Some(Self::Trace)
        } else if v.eq_ignore_ascii_case("debug") {
            Some(Self::Debug)
        } else if v.eq_ignore_ascii_case("info") {
            Some(Self::Info)
        } else if v.eq_ignore_ascii_case("warn") {
            Some(Self::Warn)
        } else if v.eq_ignore_ascii_case("error") {
            Some(Self::Error)
        } else {
            None
        }
    }
}

/// Setup tracing subscriber.
///
/// Honors `RUST_LOG` if set, otherwise uses the provided level string.
/// When `no_color` is true, ANSI escape codes are disabled.
pub fn setup_tracing(level: &str, no_color: bool) {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        tracing_subscriber::EnvFilter::from_default_env()
    } else {
        tracing_subscriber::EnvFilter::new(level)
    };

    tracing_subscriber::fmt()
        .without_time()
        .with_target(false)
        .with_ansi(!no_color)
        .with_env_filter(filter)
        .init();
}

/// Setup tracing from a `LogLevel` value.
pub fn setup_tracing_from_level(level: LogLevel, no_color: bool) {
    setup_tracing(level.as_str(), no_color);
}
