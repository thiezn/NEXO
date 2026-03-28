use std::time::Instant;

use anyhow::{bail, Context};
use tokio::process::Command;

use super::base::{GeneratedImage, ImagineRequest, ImagineResponse};

pub(super) async fn generate(
    cli_path: &str,
    output_dir: &str,
    req: ImagineRequest,
) -> anyhow::Result<ImagineResponse> {
    tokio::fs::create_dir_all(output_dir)
        .await
        .context("failed to create image output directory")?;

    let start = Instant::now();

    let output = Command::new(cli_path)
        .arg("generate")
        .args(["-p", &req.prompt])
        .args(["-s", &req.steps.to_string()])
        .args(["--seed", &req.seed.to_string()])
        .args(["--outdir", output_dir])
        .args(["--cfg-scale", &req.guidance.to_string()])
        .args(["--num-images", &req.batch_size.to_string()])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("failed to spawn qwen-image-mps")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("qwen-image-mps exited with {}: {stderr}", output.status);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut images = Vec::new();

    for line in stdout.lines() {
        let path = line.trim();
        if path.is_empty() {
            continue;
        }

        let index = images.len() as u32;
        let data = tokio::fs::read(path)
            .await
            .with_context(|| format!("failed to read generated image at {path}"))?;

        images.push(GeneratedImage {
            data,
            width: req.width,
            height: req.height,
            index,
        });
    }

    let elapsed = start.elapsed();

    Ok(ImagineResponse {
        images,
        seed_used: req.seed,
        inference_time_ms: elapsed.as_millis() as u64,
    })
}
