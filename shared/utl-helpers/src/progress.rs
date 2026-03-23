pub use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Style for an overall (top-level) progress bar.
pub fn overall_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold.cyan} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("━╸─")
}

/// Style for a child / item-level progress bar (determinate).
pub fn item_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {prefix:.bold.green} [{bar:30.green/dim}] {pos}/{len} {msg}",
    )
    .unwrap()
    .progress_chars("━╸─")
}

/// Style for a spinner (indeterminate progress).
pub fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("  {prefix:.bold.yellow} {spinner:.yellow} {msg}").unwrap()
}

/// Create an item-level spinner (indeterminate) inside a MultiProgress.
pub fn item_spinner(mp: &MultiProgress, prefix: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new_spinner());
    pb.set_style(spinner_style());
    pb.set_prefix(prefix.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}
