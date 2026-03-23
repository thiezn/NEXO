use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(
    name = "text_to_speech",
    about = "Synthesize speech from text using local inference models"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Log level
    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Synthesize speech from text
    Generate {
        /// The text to speak
        #[arg(value_name = "TEXT")]
        text: String,

        /// Voice description (e.g. "A female speaker with a warm, soft voice...")
        #[arg(short, long)]
        description: Option<String>,

        /// Model name (e.g. parler-mini, parler-large)
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum generation tokens (controls audio length)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Sampling temperature
        #[arg(long)]
        temperature: Option<f64>,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Output file path (.wav)
        #[arg(short, long)]
        output: Option<String>,
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
