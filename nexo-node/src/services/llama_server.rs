use std::path::PathBuf;
use std::time::Duration;
use tokio::process::{Child, Command};
use tracing::info;

use crate::download::paths::nexo_home_dir;

const LLAMA_SERVER_RELATIVE: &str = "inference_services/llama_cpp/llama-server";
pub(super) const LLAMA_PORT: u16 = 8001;
const LLAMA_CTX_SIZE: &str = "8192";
const LLAMA_GPU_LAYERS: &str = "99";
const LLAMA_HOST: &str = "127.0.0.1";

const INSTALL_MSG: &str = "\
llama-server binary not found.

To install:
  1. Clone the latest llama.cpp repository from https://github.com/ggml-org/llama.cpp.git
  2. build and copy the binaries to the expected location:
  `cmake -B build && cmake --build build --config Release && cp build/bin/* ~/.nexo/inference_services/llama_cpp/`
  1. Download macOS binaries from https://github.com/ggml-org/llama.cpp/releases
  2. chmod +x ~/.nexo/inference_services/llama_cpp/llama-server
  3. xattr -d com.apple.quarantine ~/.nexo/inference_services/llama_cpp/llama-server
  4. codesign --force --sign - --preserve-metadata=entitlements ~/.nexo/inference_services/llama_cpp/llama-server
  3. Re-run: nexo-node start";

pub struct LlamaServer {
    binary_path: PathBuf,
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    child: Option<Child>,
    /// Shared client reused across all health checks and requests.
    pub(super) http: reqwest::Client,
}

impl LlamaServer {
    /// Resolve paths and verify the binary exists.
    /// Returns Err with install instructions if the binary is absent.
    pub fn new(model_path: PathBuf, mmproj_path: Option<PathBuf>) -> anyhow::Result<Self> {
        let binary_path = nexo_home_dir().join(LLAMA_SERVER_RELATIVE);

        if !binary_path.exists() {
            println!("{INSTALL_MSG}");
            println!("\nExpected location: {}", binary_path.display());
            anyhow::bail!("llama-server binary not found at {}", binary_path.display());
        }

        Ok(Self {
            binary_path,
            model_path,
            mmproj_path,
            child: None,
            http: reqwest::Client::new(),
        })
    }

    /// Start llama-server as a subprocess.
    pub async fn start(&mut self) -> anyhow::Result<()> {
        info!(
            "Starting llama-server on port {LLAMA_PORT} with model {}{}",
            self.model_path.display(),
            if self.mmproj_path.is_some() { " (vision enabled)" } else { "" }
        );

        let mut args: Vec<String> = vec![
            "--model".into(),
            self.model_path.to_string_lossy().into_owned(),
            "--port".into(),
            LLAMA_PORT.to_string(),
            "--ctx-size".into(),
            LLAMA_CTX_SIZE.into(),
            "--n-gpu-layers".into(),
            LLAMA_GPU_LAYERS.into(),
            "--host".into(),
            LLAMA_HOST.into(),
        ];

        if let Some(ref mmproj) = self.mmproj_path {
            args.push("--mmproj".into());
            args.push(mmproj.to_string_lossy().into_owned());
        }

        let child = Command::new(&self.binary_path).args(&args).spawn()?;

        self.child = Some(child);
        Ok(())
    }

    /// Poll health check with backoff until healthy or timeout exceeded.
    pub async fn wait_until_healthy(&self, timeout_secs: u64) -> anyhow::Result<()> {
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
        let mut delay = Duration::from_secs(1);

        while std::time::Instant::now() < deadline {
            if check_health(&self.http).await {
                info!("llama-server is healthy on port {LLAMA_PORT}");
                return Ok(());
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(10));
        }

        anyhow::bail!(
            "llama-server did not become healthy within {timeout_secs}s on port {LLAMA_PORT}"
        )
    }

    /// Kill the subprocess if running.
    pub async fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
            info!("llama-server stopped");
        }
    }

    /// Returns true if the subprocess is still running.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            None => false,
            Some(child) => matches!(child.try_wait(), Ok(None)),
        }
    }
}

/// GET http://127.0.0.1:{LLAMA_PORT}/health — returns true if 200 OK.
/// Accepts an external client so callers can check health without holding a lock.
pub async fn check_health(client: &reqwest::Client) -> bool {
    let url = format!("http://127.0.0.1:{LLAMA_PORT}/health");
    client
        .get(&url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        // Best-effort SIGTERM on drop; explicit stop() is the primary shutdown path.
        if let Some(child) = &mut self.child
            && let Some(id) = child.id()
        {
            unsafe {
                libc::kill(id as libc::pid_t, libc::SIGTERM);
            }
        }
    }
}

#[allow(non_camel_case_types)]
mod libc {
    pub type pid_t = i32;
    pub const SIGTERM: i32 = 15;

    unsafe extern "C" {
        pub fn kill(pid: pid_t, sig: i32) -> i32;
    }
}
