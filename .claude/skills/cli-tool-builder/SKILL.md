---
name: cli-tool-builder
description: Use when creating a new CLI tool under tools/ in the nexo workspace. Covers project scaffolding, utl-helpers integration, and project conventions.
---

# CLI Tool Builder

## When to use

When building a new Rust CLI tool that will live under `tools/<tool_name>/`.

## Scaffolding checklist

1. Create `tools/<tool_name>/` with `src/main.rs`, `src/cli.rs`, and `Cargo.toml`
2. Add the crate to the workspace `members` list in the root `Cargo.toml`
3. Add domain-specific modules as needed (`src/lib.rs`, etc.)

## Cargo.toml template

```toml
[package]
name = "<tool_name>"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "<tool_name>"
path = "src/main.rs"

[dependencies]
utl-helpers = { path = "../../shared/utl-helpers" }
clap = { version = "4", features = ["derive"] }
tracing = "0"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
dirs = "6"
# Add domain-specific deps here

[lints]
workspace = true
```

Common additional deps:
- `tokio = { version = "1", features = ["rt-multi-thread", "macros"] }` — async runtime
- `serde_json = "1"` — JSON parsing (configs, API responses)
- `anyhow = "1"` — ergonomic error handling (prefer over `utl_helpers::Error` for inference tools)
- `dirs = "6"` — home directory detection for config paths

Enable utl-helpers features as needed:

- `features = ["config"]` — for TOML config load/save
- `features = ["config", "output"]` — for config + JSON/Markdown output formatting
- Default already includes `output`

Pin deps to **major version only** (e.g. `tokio = "1"`, not `"1.38.0"`).

## main.rs pattern

### Synchronous

```rust
mod cli;

use clap::Parser;
use cli::Cli;

fn main() {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);
    if let Err(e) = run(&cli) {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> utl_helpers::Result {
    // ...
    Ok(())
}
```

### Async (with tokio)

```rust
mod cli;

use clap::Parser;
use cli::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);
    if let Err(e) = run(&cli).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
```

Add `tokio = { version = "1", features = ["rt-multi-thread", "macros"] }` to deps.

## cli.rs pattern

### Simple (no subcommands)

```rust
use clap::Parser;
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(name = "<tool_name>", about = "<description>")]
pub struct Cli {
    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[arg(long, global = true)]
    pub no_color: bool,

    // tool-specific args...
}
```

### With subcommands

```rust
use clap::{Parser, Subcommand};
use utl_helpers::LogLevel;

#[derive(Parser, Debug)]
#[command(name = "<tool_name>", about = "<description>")]
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
    /// Do something
    DoThing { /* args */ },
}
```

## utl-helpers quick reference

### Always available (no feature flags)

| Function | Purpose |
|---|---|
| `utl_helpers::setup_tracing_from_level(level, no_color)` | Init tracing subscriber |
| `utl_helpers::resolve_path(&PathBuf)` | Resolve `~/`, relative, absolute paths |
| `utl_helpers::resolve_path_str(&str)` | Same but from `&str` |
| `utl_helpers::Error` | Enum: `Config`, `Io`, `Network`, `Other` |
| `utl_helpers::Result<T>` | Alias for `std::result::Result<T, Error>` (default `T = ()`) |

### `config` feature

Config files go to `~/.nexo/<tool_name>.toml`. The config struct must impl `Serialize + DeserializeOwned + Default`.

```rust
use utl_helpers::config;

let cfg: MyConfig = config::load(&path)?;           // returns Default if missing
let cfg: MyConfig = config::load_or_create(&path)?;  // creates file if missing
config::save(&cfg, &path)?;
```

### `output` feature

```rust
use utl_helpers::{OutputFormat, write_output};

// OutputFormat is a clap ValueEnum: Json, Markdown
// write_output(data, format, output_file, fields, to_markdown_fn)
// fields supports paths like "results[*].name" for filtering JSON output
```

## Config pattern

Tools that need persistent configuration use a standard AppConfig pattern:

```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub default_model: String,
    // tool-specific fields...
}

impl Default for AppConfig { /* sensible defaults */ }

impl AppConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("<tool_name>.toml")
    }

    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }
}
```

Requires `utl-helpers` with `features = ["config"]` and `dirs = "6"`.

## Error handling

Use `utl_helpers::Error` variants for typed errors:

```rust
Err(utl_helpers::Error::Io(format!("Failed to read: {}", path.display())))
Err(utl_helpers::Error::Other("something went wrong".into()))
```

`From<io::Error>`, `From<String>`, `From<&str>`, and `From<serde_json::Error>` are implemented, so `?` works naturally for those types. For other error types (e.g. `anyhow`), either convert manually or use a different error type in your `run()` function.

For inference tools that depend on `candle-*` crates, prefer `anyhow::Result` throughout (candle uses its own error type that doesn't convert to `utl_helpers::Error`).

## Pull / List / Domain dispatch pattern

Inference tools that download models follow a standard three-subcommand pattern. The domain-specific subcommand (e.g. `Describe`, `Generate`, `Transcribe`) is the primary action; `Pull` and `List` manage model assets.

```rust
async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::DomainAction { /* args */ } => {
            let app_config = AppConfig::load()?;
            // build domain config from CLI args + app_config defaults
            // call high-level API function from lib.rs
            // print result to stdout or write to --output path
        }
        Command::Pull { model } => cmd_pull(&model).await?,
        Command::List => cmd_list()?,
    }
    Ok(())
}
```

`cmd_pull` downloads files via `pull_model(manifest)`, then persists paths into `AppConfig` via `ModelPaths::from_downloads()` + `to_model_config()`.

`cmd_list` iterates `known_manifests()`, shows install status from `AppConfig`, and prints download sizes.

## Key conventions

- Binary name matches crate name
- `[lints] workspace = true` inherits clippy lints (unwrap_used, expect_used, panic = warn)
- Use `tracing::info!`, `tracing::warn!`, `tracing::error!` for logging (never `println!` for status)
- `println!` only for primary program output (e.g. listing items, final results to stdout)
- Domain logic belongs in `src/lib.rs` or dedicated modules, not in `main.rs`
- For inference tools, use `anyhow::Result` throughout (candle errors don't convert to `utl_helpers::Error`)
