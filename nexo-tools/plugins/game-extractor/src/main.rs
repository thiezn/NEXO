use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use utl_helpers::LogLevel;

use game_extractor::analyze;
use game_extractor::extractor::{Engine, ExtractionSummary, common::progress, sci, scumm};
use game_extractor::extractor::common::progress::{MultiProgress, ProgressBar};

#[derive(Parser)]
#[command(name = "game_extractor", about = "Extract and analyze assets from adventure game files")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, global = true)]
    log_level: LogLevel,

    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Extract assets from game files
    Extract(ExtractArgs),
    /// Analyze extracted images using a local LLM to generate LoRA training descriptions
    Analyze(analyze::AnalyzeArgs),
}

#[derive(clap::Args)]
struct ExtractArgs {
    /// Paths to game directories or parent directories containing multiple games
    game_dirs: Vec<PathBuf>,

    /// Output directory for extracted assets
    #[arg(short, long, default_value = "datasets/games")]
    output: PathBuf,

    /// Run analyze after extraction completes
    #[arg(long)]
    analyze: bool,

    /// Model identifier for analysis (required when --analyze is set)
    #[arg(long)]
    model: Option<String>,

    /// LM Studio API endpoint for analysis
    #[arg(long, default_value = "http://127.0.0.1:1234")]
    endpoint: String,

    /// Number of images to process in parallel during analysis
    #[arg(long, default_value_t = 4)]
    parallel: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    utl_helpers::setup_tracing_from_level(cli.log_level, cli.no_color);
    match cli.command {
        Command::Extract(args) => run_extract(args).await,
        Command::Analyze(args) => analyze::run(&args).await,
    }
}

async fn run_extract(args: ExtractArgs) -> Result<()> {
    if args.game_dirs.is_empty() {
        anyhow::bail!("At least one game directory path is required");
    }

    // Discover games from all provided paths
    let mut games: Vec<Box<dyn Engine>> = Vec::new();
    for dir in &args.game_dirs {
        games.extend(discover_games(dir));
    }

    if games.is_empty() {
        let dirs: Vec<_> = args.game_dirs.iter().map(|d| d.display().to_string()).collect();
        anyhow::bail!("No games found in {}", dirs.join(", "));
    }

    let mp = MultiProgress::new();

    let overall = mp.add(ProgressBar::new(games.len() as u64));
    overall.set_style(progress::overall_style());
    overall.set_prefix("Extracting");
    overall.set_message(format!("{} game(s)", games.len()));

    // Extract games (parallel if multiple)
    let summaries: Vec<ExtractionSummary> = if games.len() == 1 {
        let game = &games[0];
        let pb = progress::game_spinner(&mp, game.game_name());
        pb.set_message("extracting...");
        match game.extract(&args.output, Some(&pb)) {
            Ok(summary) => {
                pb.finish_and_clear();
                overall.inc(1);
                vec![summary]
            }
            Err(e) => {
                pb.finish_with_message(format!("error: {}", e));
                overall.inc(1);
                vec![]
            }
        }
    } else {
        let results: Vec<_> = games.par_iter().map(|game| {
            let pb = progress::game_spinner(&mp, game.game_name());
            pb.set_message("extracting...");
            let result = match game.extract(&args.output, Some(&pb)) {
                Ok(summary) => {
                    pb.finish_and_clear();
                    Some(summary)
                }
                Err(e) => {
                    pb.finish_with_message(format!("error: {}", e));
                    None
                }
            };
            overall.inc(1);
            result
        }).collect();

        results.into_iter().flatten().collect()
    };

    overall.finish_and_clear();

    // Print summary
    if !summaries.is_empty() {
        println!("\n=== Summary ===");
        for s in &summaries {
            let mut parts = Vec::new();
            if s.backgrounds > 0 { parts.push(format!("{} backgrounds", s.backgrounds)); }
            if s.objects > 0 { parts.push(format!("{} objects", s.objects)); }
            if s.sounds > 0 { parts.push(format!("{} sounds", s.sounds)); }
            if s.sprites > 0 { parts.push(format!("{} sprites", s.sprites)); }
            if s.speech_files > 0 { parts.push(format!("{} speech files", s.speech_files)); }
            if parts.is_empty() {
                println!("  {}: no assets extracted", s.game_name);
            } else {
                println!("  {}: {}", s.game_name, parts.join(", "));
            }
        }
    }

    // Run analysis if requested
    if args.analyze {
        println!("\n=== Running analysis ===");
        let analyze_args = analyze::AnalyzeArgs {
            paths: vec![args.output.clone()],
            endpoint: args.endpoint,
            model: args.model.unwrap_or_else(|| analyze::DEFAULT_MODEL.into()),
            parallel: args.parallel,
            output_mode: analyze::OutputMode::Review,
            force: false,
        };
        analyze::run(&analyze_args).await?;
    }

    println!("\nDone!");
    Ok(())
}

/// Try to detect a game in the given directory using all known engines.
fn try_detect(dir: &Path) -> Option<Box<dyn Engine>> {
    if let Some(e) = scumm::ScummEngine::detect(dir) {
        return Some(e);
    }
    if let Some(e) = sci::SciEngine::detect(dir) {
        return Some(e);
    }
    None
}

/// Recursively discover games in a directory tree.
/// When a game is found in a folder, don't search deeper within it.
fn discover_games(dir: &Path) -> Vec<Box<dyn Engine>> {
    // First, try detecting a game directly in this directory
    if let Some(engine) = try_detect(dir) {
        return vec![engine];
    }

    // If no game found directly, recurse into subdirectories
    let mut games = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return games;
    };

    let mut subdirs: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    subdirs.sort_by_key(|e| e.file_name());

    for entry in subdirs {
        let subdir = entry.path();
        if let Some(engine) = try_detect(&subdir) {
            games.push(engine);
        } else {
            games.extend(discover_games(&subdir));
        }
    }

    games
}
