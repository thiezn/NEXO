pub mod llama_server;

use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use llama_server::{LlamaServer, check_health};

pub struct ServiceManager {
    llama: Arc<Mutex<LlamaServer>>,
    _monitor: JoinHandle<()>,
}

impl ServiceManager {
    /// Start llama-server with the given model, wait until healthy, then
    /// spawn a background monitor that restarts it on crashes.
    pub async fn start(model_path: PathBuf) -> anyhow::Result<Self> {
        let mut server = LlamaServer::new(model_path)?;
        server.start().await?;
        server.wait_until_healthy(60).await?;

        let llama = Arc::new(Mutex::new(server));
        let monitor = spawn_monitor(Arc::downgrade(&llama));

        Ok(Self {
            llama,
            _monitor: monitor,
        })
    }

    pub async fn stop(&self) {
        self.llama.lock().await.stop().await;
    }

    pub async fn llama_healthy(&self) -> bool {
        let client = self.llama.lock().await.http.clone();
        check_health(&client).await
    }
}

impl Drop for ServiceManager {
    fn drop(&mut self) {
        self._monitor.abort();
    }
}

fn spawn_monitor(weak: Weak<Mutex<LlamaServer>>) -> JoinHandle<()> {
    tokio::spawn(async move {
        const POLL_INTERVAL: Duration = Duration::from_secs(5);
        const FAIL_THRESHOLD: u32 = 3;
        let mut consecutive_failures: u32 = 0;
        let mut backoff = Duration::from_secs(5);

        loop {
            tokio::time::sleep(POLL_INTERVAL).await;

            let Some(arc) = weak.upgrade() else {
                break;
            };

            // Clone the client without holding the lock during the HTTP call.
            let client = arc.lock().await.http.clone();
            let healthy = check_health(&client).await;

            if healthy {
                consecutive_failures = 0;
                backoff = Duration::from_secs(5);
            } else {
                consecutive_failures += 1;
                warn!(
                    "llama-server health check failed ({consecutive_failures}/{FAIL_THRESHOLD})"
                );

                if consecutive_failures >= FAIL_THRESHOLD {
                    consecutive_failures = 0;
                    warn!("llama-server appears crashed — restarting (backoff: {backoff:?})");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(60));

                    let mut server = arc.lock().await;
                    server.stop().await;
                    if let Err(e) = server.start().await {
                        warn!("failed to restart llama-server: {e}");
                    } else {
                        info!("llama-server restarted");
                    }
                }
            }
        }
    })
}
