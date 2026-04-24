mod api;
mod dataset;
mod prompt;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::extractor::common::progress::{MultiProgress, ProgressBar};
use anyhow::Result;
use clap::ValueEnum;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::extractor::common::metadata::LoraEntry;
use crate::extractor::common::progress;

pub const DEFAULT_MODEL: &str = "google/gemma-3-4b";

#[derive(Clone, ValueEnum)]
pub enum OutputMode {
    Update,
    Review,
}

#[derive(clap::Args)]
pub struct AnalyzeArgs {
    /// Paths to search for lora_training subdatasets (extracted game directories,
    /// parent directories, or specific subdataset folders)
    pub paths: Vec<PathBuf>,

    /// LM Studio API endpoint
    #[arg(long, default_value = "http://127.0.0.1:1234")]
    pub endpoint: String,

    /// Model identifier to use
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Number of images to process in parallel
    #[arg(short, long, default_value_t = 4)]
    pub parallel: usize,

    /// Output mode: "update" modifies metadata.jsonl, "review" writes metadata_review.jsonl
    #[arg(long, default_value = "review")]
    pub output_mode: OutputMode,

    /// Force re-labelling of images that already have descriptions
    #[arg(long)]
    pub force: bool,
}

pub async fn run(args: &AnalyzeArgs) -> Result<()> {
    if args.paths.is_empty() {
        anyhow::bail!("At least one path is required");
    }

    let subdatasets = discover_subdatasets(&args.paths);

    if subdatasets.is_empty() {
        let dirs: Vec<_> = args.paths.iter().map(|d| d.display().to_string()).collect();
        anyhow::bail!("No lora_training subdatasets found in {}", dirs.join(", "));
    }

    let mp = MultiProgress::new();
    let overall = mp.add(ProgressBar::new(subdatasets.len() as u64));
    overall.set_style(progress::overall_style());
    overall.set_prefix("Analyzing");
    overall.set_message(format!("{} subdataset(s)", subdatasets.len()));

    let client = Arc::new(api::LmClient::new(&args.endpoint, &args.model));

    for subdataset in &subdatasets {
        let name = subdataset
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        overall.set_message(name.clone());
        if let Err(e) = process_subdataset(
            &client,
            subdataset,
            args.parallel,
            &args.output_mode,
            args.force,
            &mp,
        )
        .await
        {
            overall.suspend(|| eprintln!("Error processing {}: {e}", subdataset.display()));
        }
        overall.inc(1);
    }

    overall.finish_and_clear();
    Ok(())
}

fn sorted_subdirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();
    dirs
}

/// Recursively discover all lora_training subdataset folders under the given paths.
/// Handles any nesting depth: parent dirs, game dirs, or specific subdataset dirs.
fn discover_subdatasets(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut results = Vec::new();
    for root in roots {
        discover_recursive(root, &mut results);
    }
    results.sort();
    results.dedup();
    results
}

fn discover_recursive(dir: &Path, results: &mut Vec<PathBuf>) {
    // If this directory itself is a subdataset (has images/ child)
    if dir.join("images").is_dir() {
        results.push(dir.to_path_buf());
        return;
    }

    // If this directory has a lora_training/ child, scan its categories
    let lora_dir = dir.join("lora_training");
    if lora_dir.is_dir() {
        for cat in sorted_subdirs(&lora_dir) {
            if cat.join("images").is_dir() {
                results.push(cat);
            }
        }
    }

    // Recurse into subdirectories (skip lora_training, already handled)
    for subdir in sorted_subdirs(dir) {
        if subdir.file_name().is_some_and(|n| n == "lora_training") {
            continue;
        }
        discover_recursive(&subdir, results);
    }
}

async fn process_subdataset(
    client: &Arc<api::LmClient>,
    subdataset_path: &Path,
    parallel: usize,
    output_mode: &OutputMode,
    force: bool,
    mp: &MultiProgress,
) -> Result<()> {
    let (images, existing) = dataset::load_dataset(subdataset_path)?;
    if images.is_empty() {
        return Ok(());
    }

    let to_process: Vec<_> = images
        .iter()
        .filter(|img| {
            let rel = format!("images/{}", img.file_name().unwrap().to_string_lossy());
            dataset::needs_labelling(existing.get(&rel), force)
        })
        .collect();

    if to_process.is_empty() {
        return Ok(());
    }

    let name = subdataset_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let pb = mp.add(ProgressBar::new(to_process.len() as u64));
    pb.set_style(progress::game_style());
    pb.set_prefix(name);

    let semaphore = Arc::new(Semaphore::new(parallel));

    let mut join_set = JoinSet::new();

    for image_path in to_process {
        let image_path = image_path.clone();
        let client = Arc::clone(client);
        let sem = Arc::clone(&semaphore);
        let pb = pb.clone();

        join_set.spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let filename = image_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            pb.set_message(filename.clone());

            let result = client.describe_image(&image_path).await;
            let rel_path = format!("images/{filename}");

            let entry = match result {
                Ok(description) => Some(LoraEntry {
                    image: rel_path,
                    text: description,
                }),
                Err(_) => None,
            };
            pb.inc(1);
            entry
        });
    }

    let mut results: Vec<LoraEntry> = Vec::new();
    while let Some(result) = join_set.join_next().await {
        if let Ok(Some(entry)) = result {
            results.push(entry);
        }
    }

    pb.finish_and_clear();

    let mut merged = existing;
    for entry in results {
        merged.insert(entry.image.clone(), entry);
    }

    let mut final_entries: Vec<LoraEntry> = merged.into_values().collect();
    final_entries.sort_by(|a, b| a.image.cmp(&b.image));

    dataset::write_metadata(&final_entries, subdataset_path, output_mode)?;

    Ok(())
}
