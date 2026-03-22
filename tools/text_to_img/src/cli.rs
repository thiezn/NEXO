use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(
    name = "text_to_img",
    about = "Generate images from text using local inference models"
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
    /// Generate images from a text prompt
    Generate {
        /// The text prompt to generate images from
        #[arg(value_name = "PROMPT")]
        prompt: String,

        /// Model name (e.g. flux-schnell:q8, z-image-turbo:q8)
        #[arg(short, long)]
        model: Option<String>,

        /// Image width in pixels
        #[arg(long)]
        width: Option<u32>,

        /// Image height in pixels
        #[arg(long)]
        height: Option<u32>,

        /// Number of denoising steps
        #[arg(long)]
        steps: Option<u32>,

        /// Guidance scale (CFG)
        #[arg(long)]
        guidance: Option<f64>,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(short = 'n', long, default_value_t = 1)]
        num_images: u32,

        /// Output directory for generated images
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
