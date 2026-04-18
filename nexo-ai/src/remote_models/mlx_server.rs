use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::process::Stdio;
use tokio::process::{Child, Command};

/// Metadata for a model reported by the mlx_vlm server.
#[derive(Debug, Clone, Deserialize)]
pub struct MlxModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<MlxModelInfo>,
}

/// Manages the lifecycle of an `mlx_vlm.server` Python process.
///
/// All public methods are async — external crates can call them directly
/// from tokio contexts without blocking.
pub struct MlxServer {
    process: Option<Child>,
    host: String,
    port: u16,
    base_url: String,
    venv_path: Option<String>,
    client: reqwest::Client,
}

impl MlxServer {
    /// Create an idle server manager. No process is started yet.
    pub fn new(host: &str, port: u16, venv_path: Option<String>) -> Self {
        Self {
            process: None,
            base_url: format!("http://{host}:{port}"),
            host: host.to_string(),
            port,
            venv_path,
            client: reqwest::Client::new(),
        }
    }

    /// Base URL of the server (e.g. `http://127.0.0.1:8080`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Start the mlx_vlm server process (without loading a model).
    ///
    /// Polls `/health` until the server is ready (up to 30 seconds).
    /// Returns an error if `mlx_vlm` is not installed.
    pub async fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let python = self.python_path();
        let port_str = self.port.to_string();

        let child = Command::new(&python)
            .args([
                "-m",
                "mlx_vlm.server",
                "--port",
                &port_str,
                "--host",
                &self.host,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start mlx_vlm server using '{python}'. \
                     Is mlx-vlm installed? Install with: pip install mlx-vlm"
                )
            })?;

        self.process = Some(child);

        // Poll /health until ready
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if tokio::time::Instant::now() > deadline {
                self.stop().await;
                bail!(
                    "mlx_vlm server did not become healthy within 30 seconds on {}:{}",
                    self.host,
                    self.port
                );
            }

            // Check the process hasn't exited
            if let Some(ref mut child) = self.process
                && let Some(status) = child.try_wait()?
            {
                bail!(
                    "mlx_vlm server exited immediately with status {status}. \
                     Is mlx-vlm installed? Install with: pip install mlx-vlm"
                );
            }

            match self.health_check().await {
                Ok(true) => {
                    tracing::info!("mlx_vlm server healthy at {}:{}", self.host, self.port);
                    return Ok(());
                }
                _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }
    }

    /// Stop the server process.
    pub async fn stop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
            tracing::info!("mlx_vlm server stopped");
        }
    }

    /// Start the server if it's not already running.
    pub async fn ensure_running(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }
        self.start().await
    }

    /// Whether the server process is still alive.
    pub fn is_running(&mut self) -> bool {
        match self.process {
            Some(ref mut child) => child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }

    /// Check the `/health` endpoint.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url());
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// List models available on the server via `GET /v1/models`.
    pub async fn list_models(&self) -> Result<Vec<MlxModelInfo>> {
        let url = format!("{}/v1/models", self.base_url());
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to query /v1/models")?;

        if !resp.status().is_success() {
            bail!("/v1/models returned status {}", resp.status());
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .context("failed to parse /v1/models response")?;
        Ok(body.data)
    }

    /// Unload the currently loaded model via `POST /unload`.
    pub async fn unload_model(&self) -> Result<()> {
        let url = format!("{}/unload", self.base_url());
        let resp = self
            .client
            .post(&url)
            .send()
            .await
            .context("failed to POST /unload")?;

        if !resp.status().is_success() {
            bail!("/unload returned status {}", resp.status());
        }
        tracing::info!("unloaded model from mlx_vlm server");
        Ok(())
    }

    fn python_path(&self) -> String {
        match &self.venv_path {
            Some(venv) => format!("{venv}/bin/python3"),
            None => "python3".to_string(),
        }
    }
}
