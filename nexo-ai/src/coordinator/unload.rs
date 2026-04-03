use crate::device;
use anyhow::Result;

impl super::Coordinator {
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        if let Some(slot) = self.slots.get_mut(model_name) {
            slot.model.unload();
            self.stats.record_model_unloaded(model_name);
            tracing::info!("unloaded model '{}'", model_name);
            Ok(())
        } else {
            anyhow::bail!("model '{}' not loaded", model_name)
        }
    }

    pub fn unload_all(&mut self) {
        let names: Vec<String> = self.slots.keys().cloned().collect();
        for slot in self.slots.values_mut() {
            if slot.is_loaded() {
                slot.model.unload();
            }
        }
        for name in &names {
            self.stats.record_model_unloaded(name);
        }
        self.slots.clear();
        self.clear_active_models();
        tracing::info!("unloaded all models");
    }

    pub fn free_memory(&mut self, bytes_needed: u64) -> Result<u64> {
        let mut freed: u64 = 0;
        let mut loaded: Vec<(String, u64)> = self
            .slots
            .iter()
            .filter(|(_, s)| s.is_loaded())
            .map(|(name, s)| (name.clone(), s.memory_estimate_bytes()))
            .collect();

        // Sort by largest memory footprint first
        loaded.sort_by(|a, b| b.1.cmp(&a.1));

        for (name, mem) in loaded {
            if freed >= bytes_needed {
                break;
            }
            if let Some(slot) = self.slots.get_mut(&name) {
                slot.model.unload();
                freed += mem;
                tracing::info!("freed {} by unloading '{}'", device::fmt_gb(mem), name);
            }
        }
        Ok(freed)
    }
}
