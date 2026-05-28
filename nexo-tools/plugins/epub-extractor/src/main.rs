mod cli;
mod epub_reader;
mod extractor;
mod models;
mod writer;

use clap::Parser;
use cli::{Cli, ImageMode};
use rayon::prelude::*;
use std::path::Path;

fn main() {
    let cli = Cli::parse();
    cli.common.init_tracing()?;

    if let Err(e) = run(&cli) {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> cli_helpers::Result {
    let output_dir = cli_helpers::resolve_path(&cli.output_dir)?;
    let input = cli_helpers::resolve_path(&cli.input)?;
    let image_mode = cli.image_mode();

    if input.is_file() {
        let dir = process_single(&input, &output_dir, image_mode)?;
        tracing::info!("Output written to {}", dir.display());
    } else if input.is_dir() {
        process_directory(&input, &output_dir, image_mode)?;
    } else {
        return Err(cli_helpers::Error::Other(format!(
            "Input path does not exist: {}",
            input.display()
        )));
    }

    Ok(())
}

fn process_single(
    epub_path: &Path,
    output_dir: &Path,
    image_mode: ImageMode,
) -> cli_helpers::Result<std::path::PathBuf> {
    let result = extractor::extract_single(epub_path, image_mode)?;
    writer::write_result(output_dir, &result, image_mode)
}

fn process_directory(
    input_dir: &Path,
    output_dir: &Path,
    image_mode: ImageMode,
) -> cli_helpers::Result {
    let epub_files: Vec<_> = std::fs::read_dir(input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("epub"))
        })
        .collect();

    if epub_files.is_empty() {
        tracing::warn!("No .epub files found in {}", input_dir.display());
        return Ok(());
    }

    tracing::info!(
        "Found {} epub file(s), processing in parallel",
        epub_files.len()
    );

    let results: Vec<_> = epub_files
        .par_iter()
        .map(|path| {
            tracing::info!("Processing: {}", path.display());
            match process_single(path, output_dir, image_mode) {
                Ok(dir) => {
                    tracing::info!("OK: {} -> {}", path.display(), dir.display());
                    Ok(dir)
                }
                Err(e) => {
                    tracing::error!("FAIL: {}: {e}", path.display());
                    Err(e)
                }
            }
        })
        .collect();

    let failed = results.iter().filter(|r| r.is_err()).count();
    tracing::info!(
        "Done: {} succeeded, {failed} failed",
        results.len() - failed
    );

    if failed > 0 {
        return Err(cli_helpers::Error::Other(format!(
            "{failed} extraction(s) failed"
        )));
    }

    Ok(())
}
