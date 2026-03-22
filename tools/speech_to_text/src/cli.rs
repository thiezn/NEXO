use clap::{Parser, Subcommand, ValueEnum};
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(
    name = "speech_to_text",
    about = "Transcribe audio to text using local Whisper models"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Transcribe an audio file to text
    Transcribe {
        /// Path to the audio file (WAV, MP3, FLAC, OGG)
        #[arg(value_name = "FILE")]
        file: String,

        /// Model name (e.g. whisper-large-v3, whisper-large-v3-turbo)
        #[arg(short, long)]
        model: Option<String>,

        /// Output format
        #[arg(short = 'f', long, value_enum, default_value_t = TranscriptFormat::Text)]
        format: TranscriptFormat,

        /// Language code (e.g. "en", "nl") or "auto" for detection
        #[arg(long, default_value = "auto")]
        language: String,

        /// Translate to English instead of transcribing
        #[arg(long)]
        translate: bool,

        /// Output file path (stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,

        /// Disable timestamps in output
        #[arg(long)]
        no_timestamps: bool,
    },

    /// Download model weights
    Pull {
        /// Model name to download, or "all" for all models
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// List available models
    List,
}

#[derive(ValueEnum, Clone, Debug, Copy)]
pub enum TranscriptFormat {
    Text,
    Srt,
    Vtt,
    Json,
}

impl From<TranscriptFormat> for speech_to_text::output::OutputFormat {
    fn from(f: TranscriptFormat) -> Self {
        match f {
            TranscriptFormat::Text => Self::Text,
            TranscriptFormat::Srt => Self::Srt,
            TranscriptFormat::Vtt => Self::Vtt,
            TranscriptFormat::Json => Self::Json,
        }
    }
}
