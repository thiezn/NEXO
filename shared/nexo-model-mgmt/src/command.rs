//! Reusable `models` CLI command.

use std::process::ExitCode;

use cli_helpers::{CommandContext, Runnable};
use comfy_table::{ContentArrangement, Table, presets::ASCII_MARKDOWN};
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
            ModelsAction::Pull { models, force } => run_pull(context, &models, force).await,
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
        /// Model names, category names, or `all`.
        #[arg(value_name = "MODEL", required = true, num_args = 1..)]
        models: Vec<String>,
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

    let mut table = Table::new();
    table.load_preset(ASCII_MARKDOWN);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header([
        "ID",
        "FAMILY",
        "BACKEND",
        "CAPABILITIES",
        "SIZE",
        "DOWNLOADED",
        "DESCRIPTION",
    ]);

    for entry in entries {
        let capabilities = entry
            .capabilities
            .iter()
            .copied()
            .map(capability_label)
            .collect::<Vec<_>>()
            .join(",");
        table.add_row([
            entry.id,
            entry.family,
            entry.backend,
            capabilities,
            format!("{:.1}G", entry.size_gb),
            if entry.is_downloaded {
                "yes".to_string()
            } else {
                "no".to_string()
            },
            entry.description,
        ]);
    }

    context.stdout_line(table.to_string())?;

    Ok(ExitCode::SUCCESS)
}

async fn run_pull(
    context: &mut CommandContext,
    models: &[String],
    force: bool,
) -> Result<ExitCode> {
    let manifests = manifests_to_pull(models)?;

    if manifests.is_empty() {
        context.stdout_line("no models matched the requested pull targets")?;
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

fn manifests_to_pull(models: &[String]) -> Result<Vec<&'static crate::manifest::ModelManifest>> {
    let mut resolved: Vec<&'static crate::manifest::ModelManifest> = Vec::new();

    for model in models {
        let manifests = manifests_for_target(model)?;
        for manifest in manifests {
            if !resolved
                .iter()
                .any(|existing| existing.id() == manifest.id())
            {
                resolved.push(manifest);
            }
        }
    }

    Ok(resolved)
}

fn manifests_for_target(model: &str) -> Result<Vec<&'static crate::manifest::ModelManifest>> {
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

#[cfg(test)]
mod tests {
    use super::manifests_to_pull;

    #[test]
    fn manifests_to_pull_deduplicates_repeated_targets() {
        let manifests = manifests_to_pull(&[
            "gemma-4-e2b-it-uqff-q4k".to_string(),
            "gemma-4-e2b-it-uqff-q4k".to_string(),
        ])
        .unwrap();

        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].id(), "gemma-4-e2b-it-uqff-q4k");
    }

    #[test]
    fn manifests_to_pull_deduplicates_category_overlap() {
        let manifests = manifests_to_pull(&["flux".to_string(), "flux.2-dev".to_string()]).unwrap();
        let flux_dev_count = manifests
            .iter()
            .filter(|manifest| manifest.id() == "flux.2-dev")
            .count();

        assert_eq!(flux_dev_count, 1);
    }
}
