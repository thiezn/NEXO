use cli_helpers::CommandContext;
use comfy_table::{ContentArrangement, Table, presets::ASCII_MARKDOWN};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nexo_ai::{
    CatalogDownloadProgress, DownloadOptions, FileDownloadProgress, ModelCatalog, ModelFileKind,
};
use nexo_core::ModelId;
use nexo_core::NodeProperties;
use nexo_node::Result;
use std::process::ExitCode;
use std::sync::Arc;

use super::node_config_path;

/// Reusable local model management command with `list` and `pull` actions.
#[derive(clap::Args, Debug, Clone)]
pub struct ModelsCommand {
    /// Model management action.
    #[command(subcommand)]
    pub action: ModelsAction,
}

impl ModelsCommand {
    /// Runs the selected model management action.
    ///
    /// # Arguments
    ///
    /// * `context` - The CLI command context used for output.
    pub async fn run(self, context: &mut CommandContext) -> Result<ExitCode> {
        match self.action {
            ModelsAction::List => run_list_command(context, ModelCatalog::new()),
            ModelsAction::Pull {
                model_ids,
                force,
                max_concurrent_files,
                keep_cache,
                proxy,
            } => {
                let mut options = DownloadOptions::default();
                options.force = force;
                if let Some(max_concurrent_files) = max_concurrent_files {
                    options.max_concurrent_files = max_concurrent_files;
                }
                options.cleanup_cache_on_success = !keep_cache;
                options.proxy = resolve_pull_proxy(proxy)?;

                run_pull_command(context, model_ids, options).await
            }
        }
    }
}

/// Subcommands for local model management.
#[derive(clap::Subcommand, Debug, Clone)]
pub enum ModelsAction {
    /// List known models and their download status.
    List,
    /// Pull one model, all models, or every model in a category.
    Pull {
        /// Force a re-download even if the target files already validate locally.
        #[arg(long)]
        force: bool,

        /// Maximum number of files to download concurrently per pull request.
        #[arg(long, value_name = "COUNT")]
        max_concurrent_files: Option<usize>,

        /// Keep successfully downloaded staged cache files instead of deleting them.
        #[arg(long)]
        keep_cache: bool,

        /// Optional proxy URL (e.g. socks5://127.0.0.1:6789). Overrides nexo-node.toml when set.
        #[arg(long, value_name = "URL")]
        proxy: Option<String>,

        /// Model IDs, category names, or `all`.
        #[arg(value_name = "MODEL", required = true, num_args = 1..)]
        model_ids: Vec<ModelId>,
    },
}

fn resolve_pull_proxy(cli_proxy: Option<String>) -> Result<Option<String>> {
    if cli_proxy.is_some() {
        return Ok(cli_proxy);
    }

    let path = node_config_path();
    if !path.exists() {
        return Ok(None);
    }

    let config: NodeProperties = cli_helpers::config::load(&path)?;
    Ok(config.proxy().map(ToOwned::to_owned))
}

/// Run the `models list` subcommand and print the results to the console.
fn run_list_command(context: &mut CommandContext, catalog: ModelCatalog) -> Result<ExitCode> {
    // let table = list_models(context, catalog)?;
    // context.print_table(&table, ContentArrangement::Dynamic, ASCII_MARKDOWN);

    let manifests = catalog.list_all_manifests();

    let mut table = Table::new();
    table.load_preset(ASCII_MARKDOWN);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header([
        "ID",
        "FAMILY",
        "CAPABILITIES",
        "RAM SIZE (GB)",
        "DOWNLOAD SIZE",
        "DOWNLOADED",
    ]);

    for manifest in manifests {
        let downloaded = if manifest.is_present_locally() {
            "Yes"
        } else {
            "No"
        };

        table.add_row([
            &manifest.model_id().to_string(),
            &manifest.family().to_string(),
            &manifest
                .capabilities()
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
            &manifest.ram_size_gb().to_string(),
            &manifest.download_size().to_string(),
            downloaded,
        ]);
    }

    context.stdout_line(table.to_string())?;

    Ok(ExitCode::SUCCESS)
}

/// Run the `models pull` subcommand and print the results to the console.
///
/// # Arguments
///
/// * `context` - The CLI command context used for output.
/// * `model_ids` - The models that should be downloaded.
/// * `options` - Download behavior options derived from CLI flags.
async fn run_pull_command(
    context: &mut CommandContext,
    model_ids: Vec<ModelId>,
    options: DownloadOptions,
) -> Result<ExitCode> {
    for model_id in &model_ids {
        context.stdout_line(format!("Pulling model: {}", model_id))?;
    }

    let catalog = ModelCatalog::new();
    let progress = Arc::new(CliDownloadProgress::new());
    catalog
        .download_models_with_options(&model_ids, options, progress)
        .await?;

    Ok(ExitCode::SUCCESS)
}

struct CliDownloadProgress {
    multi: MultiProgress,
    style: ProgressStyle,
}

impl CliDownloadProgress {
    /// Creates a CLI progress renderer backed by `indicatif::MultiProgress`.
    fn new() -> Self {
        let style = ProgressStyle::with_template(
            "  {msg:<48} [{bar:30.cyan/dim}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("#>-");

        Self {
            multi: MultiProgress::with_draw_target(ProgressDrawTarget::stderr()),
            style,
        }
    }
}

impl CatalogDownloadProgress for CliDownloadProgress {
    fn model_started(&self, model_id: &ModelId, file_count: usize, total_bytes: u64) {
        let _ = self.multi.println(format!(
            "Downloading {model_id} ({file_count} files, {total_bytes} bytes)"
        ));
    }

    fn file_started(
        &self,
        model_id: &ModelId,
        kind: ModelFileKind,
        _repo: &str,
        remote_path: &str,
        size_bytes: u64,
    ) -> Arc<dyn FileDownloadProgress> {
        let bar = self.multi.add(ProgressBar::new(size_bytes));
        bar.set_style(self.style.clone());
        bar.set_message(format!("{model_id} [{kind:?}] {remote_path}"));
        Arc::new(IndicatifFileDownloadProgress { bar })
    }

    fn model_finished(&self, model_id: &ModelId) {
        let _ = self.multi.println(format!("Finished {model_id}"));
    }
}

struct IndicatifFileDownloadProgress {
    bar: ProgressBar,
}

impl FileDownloadProgress for IndicatifFileDownloadProgress {
    /// Initializes the progress bar for a single file download.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - The total expected file size in bytes.
    /// * `label` - The label shown for the file progress bar.
    fn init(&self, size_bytes: u64, label: &str) {
        self.bar.set_length(size_bytes);
        self.bar.set_message(label.to_string());
    }

    /// Advances the progress bar by the downloaded byte delta.
    ///
    /// # Arguments
    ///
    /// * `delta_bytes` - The number of bytes downloaded since the previous update.
    fn advance(&self, delta_bytes: u64) {
        self.bar.inc(delta_bytes);
    }

    /// Marks the progress bar as finished.
    fn finish(&self) {
        self.bar.finish_with_message("done");
    }
}
