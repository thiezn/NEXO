//! Reusable `models` CLI command.

use std::process::ExitCode;

use cli_helpers::{CommandContext, Runnable};
use nexo_core::{ModelCapability, SupportedModality};

use crate::registry::{
    capability_label, find_manifest, known_manifests, list_models, manifests_for_capability,
    manifests_for_modality,
};
use crate::{Error, Result, pull_model};

/// Reusable local model management command with `list` and `pull` actions.
#[derive(clap::Args, Debug, Clone)]
pub struct ModelsCommand {
    /// Model management action.
    #[command(subcommand)]
    pub action: ModelsAction,
}

impl ModelsCommand {
    /// Creates a new `models` command wrapper.
    #[must_use]
    pub const fn new(action: ModelsAction) -> Self {
        Self { action }
    }

    /// Runs the command inside an existing async runtime.
    pub async fn run_async(self, context: &mut CommandContext) -> Result<ExitCode> {
        match self.action {
            ModelsAction::List => run_list(context),
            ModelsAction::Pull { model, force } => run_pull(context, &model, force).await,
        }
    }
}

impl Runnable for ModelsCommand {
    type Error = crate::Error;

    fn run(self, context: &mut CommandContext) -> Result<ExitCode> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(Error::Runtime)?;
        runtime.block_on(self.run_async(context))
    }
}

/// Subcommands for local model management.
#[derive(clap::Subcommand, Debug, Clone)]
pub enum ModelsAction {
    /// List known models and their download status.
    List,
    /// Pull one model, all models, or every model in a category.
    Pull {
        /// Model name, category name, or `all`.
        #[arg(value_name = "MODEL")]
        model: String,
        /// Replace existing files even when size checks pass.
        #[arg(long)]
        force: bool,
    },
}

fn run_list(context: &mut CommandContext) -> Result<ExitCode> {
    let entries = list_models();

    if entries.is_empty() {
        context.stdout_line("no models registered")?;
        return Ok(ExitCode::SUCCESS);
    }

    context.stdout_line(format!(
        "{:<34} {:<14} {:<20} {:<26} {:<8} {:<12} DESCRIPTION",
        "ID", "FAMILY", "BACKEND", "CAPABILITIES", "SIZE", "DOWNLOADED"
    ))?;
    context.stdout_line("-".repeat(132))?;

    for entry in entries {
        let capabilities = entry
            .capabilities
            .iter()
            .copied()
            .map(capability_label)
            .collect::<Vec<_>>()
            .join(",");
        context.stdout_line(format!(
            "{:<34} {:<14} {:<20} {:<26} {:<8} {:<12} {}",
            entry.id,
            entry.family,
            entry.backend,
            capabilities,
            format!("{:.1}G", entry.size_gb),
            if entry.is_downloaded { "yes" } else { "no" },
            entry.description
        ))?;
    }

    Ok(ExitCode::SUCCESS)
}

async fn run_pull(context: &mut CommandContext, model: &str, force: bool) -> Result<ExitCode> {
    let manifests = manifests_to_pull(model)?;

    if manifests.is_empty() {
        context.stdout_line(format!("no models found for '{model}'"))?;
        return Ok(ExitCode::SUCCESS);
    }

    for manifest in manifests {
        context.stdout_line(format!(
            "pulling {} ({:.1} GB)...",
            manifest.id(),
            manifest.size_gb
        ))?;
        let downloads = pull_model(manifest, force).await?;
        context.stdout_line(format!(
            "  downloaded {} files for {}",
            downloads.len(),
            manifest.id()
        ))?;
    }

    context.stdout_line("done")?;
    Ok(ExitCode::SUCCESS)
}

fn manifests_to_pull(model: &str) -> Result<Vec<&'static crate::manifest::ModelManifest>> {
    if model == "all" {
        return Ok(known_manifests().iter().collect());
    }

    if let Some(manifests) = manifests_for_query(model) {
        return Ok(manifests);
    }

    if let Some(manifest) = find_manifest(model) {
        return Ok(vec![manifest]);
    }

    Err(Error::UnknownModel {
        model: model.to_string(),
        known: known_manifests()
            .iter()
            .map(|manifest| manifest.id().to_string())
            .collect(),
    })
}

fn manifests_for_query(query: &str) -> Option<Vec<&'static crate::manifest::ModelManifest>> {
    let query = query.trim().to_ascii_lowercase();
    match query.as_str() {
        "chat" | "text" => Some(manifests_for_capability(ModelCapability::TextGeneration)),
        "tool" | "tools" => Some(manifests_for_capability(ModelCapability::ToolCalling)),
        "embedding" | "embeddings" => Some(manifests_for_capability(ModelCapability::Embeddings)),
        "image" | "vision" => Some(merge_manifests(
            manifests_for_modality(SupportedModality::Image),
            manifests_for_capability(ModelCapability::ImageGeneration),
        )),
        "listen" | "audio" | "speech" | "stt" | "asr" => {
            Some(manifests_for_modality(SupportedModality::Audio))
        }
        "flux" | "image-generation" => {
            Some(manifests_for_capability(ModelCapability::ImageGeneration))
        }
        _ => None,
    }
}

fn merge_manifests(
    mut left: Vec<&'static crate::manifest::ModelManifest>,
    right: Vec<&'static crate::manifest::ModelManifest>,
) -> Vec<&'static crate::manifest::ModelManifest> {
    for manifest in right {
        if !left.iter().any(|existing| existing.id() == manifest.id()) {
            left.push(manifest);
        }
    }
    left
}
