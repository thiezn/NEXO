use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(
    name = "multimodal",
    about = "Multimodal inference using local Qwen3.5 models"
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
    /// Describe an image or ask a question about it
    Describe {
        /// Path to the image file (PNG, JPEG, WebP)
        #[arg(value_name = "IMAGE")]
        image: String,

        /// Prompt/question about the image
        #[arg(short, long, default_value = "Describe this image in detail.")]
        prompt: String,

        /// Model name (e.g. qwen3.5-9b)
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy)
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,

        /// Top-p (nucleus) sampling threshold
        #[arg(long, default_value_t = 0.9)]
        top_p: f64,

        /// Output file path (stdout if not specified)
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
