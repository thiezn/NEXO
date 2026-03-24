use clap::{Parser, Subcommand, ValueEnum};
use utl_helpers::LogLevel;

#[derive(Parser)]
#[command(name = "nexo", about = "NEXO Gateway - Neural Extension Operator")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize gateway configuration and database
    Init,

    /// Start the gateway WebSocket server
    Start {
        /// Override bind host
        #[arg(long)]
        host: Option<String>,

        /// Override bind port
        #[arg(long)]
        port: Option<u16>,
    },

    /// Generate JSON schemas for the WebSocket protocol
    Schema {
        /// Section to generate
        #[arg(value_enum, default_value_t = SchemaTarget::All)]
        section: SchemaTarget,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SchemaTarget {
    All,
    Frames,
    Connect,
    Methods,
    Events,
    Errors,
}
