use clap::Parser;
use std::path::PathBuf;
use utl_helpers::LogLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    /// Embed base64-encoded image data in the JSON output (default)
    Base64,
    /// Write images as separate files in an images/ directory
    Files,
    /// Skip image extraction entirely
    None,
}

#[derive(Parser, Debug)]
#[command(
    name = "extractor_epub",
    about = "Extract EPUB books into structured JSON and images"
)]
pub struct Cli {
    /// Path to a single .epub file or a directory containing .epub files
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Base directory for output
    #[arg(short, long, default_value = "./datasets/books")]
    pub output_dir: PathBuf,

    /// Skip image extraction entirely
    #[arg(long, conflicts_with = "images_as_files")]
    pub no_images: bool,

    /// Write images as separate files instead of embedding base64 in JSON
    #[arg(long)]
    pub images_as_files: bool,

    /// Log level
    #[arg(short, long, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

impl Cli {
    pub fn image_mode(&self) -> ImageMode {
        if self.no_images {
            ImageMode::None
        } else if self.images_as_files {
            ImageMode::Files
        } else {
            ImageMode::Base64
        }
    }
}
