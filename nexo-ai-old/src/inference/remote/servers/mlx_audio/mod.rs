use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::inference::remote::openai::model::OpenAiServerControl;
use crate::inference::remote::openai::protocol::{OpenAiModelInfo, OpenAiModelsResponse};

const STARTUP_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, Deserialize)]
pub struct MlxAudioRootInfo {
    pub message: String,
}

/// Manages the lifecycle of an `mlx_audio.server` Python process.
pub struct MlxAudioServer {
    process: Option<Child>,
    host: String,
    port: u16,
    base_url: String,
    venv_path: Option<String>,
    hf_endpoint: String,
    client: reqwest::Client,
}

impl MlxAudioServer {
    pub fn new(host: &str, port: u16, venv_path: Option<String>, hf_endpoint: String) -> Self {
        Self {
            process: None,
            base_url: format!("http://{host}:{port}"),
            host: host.to_string(),
            port,
            venv_path,
            hf_endpoint,
            client: reqwest::Client::new(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub async fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let python = self.python_path();
        let port_str = self.port.to_string();

        let child = Command::new(&python)
            .args([
                "-m",
                "mlx_audio.server",
                "--port",
                &port_str,
                "--host",
                &self.host,
            ])
            .env("HF_ENDPOINT", &self.hf_endpoint)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start mlx_audio server using '{python}'. Is mlx-audio installed? Install with: pip install mlx-audio"
                )
            })?;

        self.process = Some(child);

        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(STARTUP_TIMEOUT_SECS);
        loop {
            if tokio::time::Instant::now() > deadline {
                self.stop().await;
                bail!(
                    "mlx_audio server did not become ready within {} seconds on {}:{}",
                    STARTUP_TIMEOUT_SECS,
                    self.host,
                    self.port
                );
            }

            if let Some(ref mut child) = self.process
                && let Some(status) = child.try_wait()?
            {
                self.process = None;
                bail!(
                    "mlx_audio server exited immediately with status {status} while starting on {}:{} using '{}'. Is mlx-audio installed? Install with: pip install mlx-audio",
                    self.host,
                    self.port,
                    python,
                );
            }

            match self.health_check().await {
                Ok(true) => {
                    tracing::info!("mlx_audio server healthy at {}:{}", self.host, self.port);
                    return Ok(());
                }
                _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }
    }

    pub async fn stop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
            tracing::info!("mlx_audio server stopped");
        }
    }

    pub async fn ensure_running(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }
        self.start().await
    }

    pub fn is_running(&mut self) -> bool {
        match self.process {
            Some(ref mut child) => child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        self.root_info().await.map(|_| true)
    }

    pub async fn root_info(&self) -> Result<MlxAudioRootInfo> {
        let url = format!("{}/", self.base_url());
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to query mlx_audio root endpoint")?;

        if !resp.status().is_success() {
            bail!("mlx_audio root endpoint returned status {}", resp.status());
        }

        resp.json()
            .await
            .context("failed to parse mlx_audio root response")
    }

    pub async fn list_models(&self) -> Result<Vec<OpenAiModelInfo>> {
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

        let body: OpenAiModelsResponse = resp
            .json()
            .await
            .context("failed to parse /v1/models response")?;
        Ok(body.data)
    }

    pub async fn unload_model(&self, model_name: &str) -> Result<()> {
        let url = format!("{}/v1/models", self.base_url());
        let resp = self
            .client
            .delete(&url)
            .query(&[("model_name", model_name)])
            .send()
            .await
            .context("failed to DELETE /v1/models")?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        if !resp.status().is_success() {
            bail!("/v1/models DELETE returned status {}", resp.status());
        }

        tracing::info!("unloaded model '{model_name}' from mlx_audio server");
        Ok(())
    }

    fn python_path(&self) -> String {
        match &self.venv_path {
            Some(venv) => format!("{venv}/bin/python3"),
            None => "python3".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct MlxAudioHandle {
    inner: Arc<Mutex<MlxAudioServer>>,
}

impl MlxAudioHandle {
    pub fn new(host: &str, port: u16, venv_path: Option<String>, hf_endpoint: String) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MlxAudioServer::new(
                host,
                port,
                venv_path,
                hf_endpoint,
            ))),
        }
    }

    pub fn shared(&self) -> Arc<Mutex<MlxAudioServer>> {
        self.inner.clone()
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
    }
}

impl OpenAiServerControl for MlxAudioHandle {
    fn ensure_running(&self) -> Result<()> {
        Self::block_on(async { self.inner.lock().await.ensure_running().await })
    }

    fn unload_model(&self, model_id: &str) -> Result<()> {
        let model_id = model_id.to_string();
        Self::block_on(async { self.inner.lock().await.unload_model(&model_id).await })
    }
}
