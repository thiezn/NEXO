use crate::config::CoordinatorConfig;

use super::{mlx_audio, mlx_vlm};

#[derive(Default)]
pub struct ManagedProviderServers {
    mlx_audio: Option<mlx_audio::MlxAudioHandle>,
    mlx_vlm: Option<mlx_vlm::MlxVlmHandle>,
}

impl ManagedProviderServers {
    pub fn mlx_audio(&mut self, config: &CoordinatorConfig) -> mlx_audio::MlxAudioHandle {
        let (host, port) = config.mlx_audio_server_addr();
        let venv = config.mlx_audio_venv_path.clone();
        let hf_endpoint = config.mlx_audio_hf_endpoint();
        self.mlx_audio
            .get_or_insert_with(|| mlx_audio::MlxAudioHandle::new(&host, port, venv, hf_endpoint))
            .clone()
    }

    pub fn mlx_vlm(&mut self, config: &CoordinatorConfig) -> mlx_vlm::MlxVlmHandle {
        let (host, port) = config.mlx_vlm_server_addr();
        let venv = config.mlx_vlm_venv_path.clone();
        self.mlx_vlm
            .get_or_insert_with(|| mlx_vlm::MlxVlmHandle::new(&host, port, venv))
            .clone()
    }

    pub fn stop_all(&self) {
        if let Some(server) = &self.mlx_audio {
            let server = server.clone();
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    server.shared().lock().await.stop().await;
                })
            });
        }

        if let Some(server) = &self.mlx_vlm {
            let server = server.clone();
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    server.shared().lock().await.stop().await;
                })
            });
        }
    }
}