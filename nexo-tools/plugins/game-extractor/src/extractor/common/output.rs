use anyhow::{Context, Result};
use image::RgbImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::metadata::LoraEntry;

pub struct OutputManager {
    base_dir: PathBuf,
    room_names: HashMap<u16, String>,
}

impl OutputManager {
    pub fn new_for_game(output_root: &Path, display_name: &str) -> Result<Self> {
        Self::new(output_root, display_name, Default::default())
    }

    pub fn new(
        output_root: &Path,
        display_name: &str,
        room_names: HashMap<u8, String>,
    ) -> Result<Self> {
        let folder_name = display_name
            .to_lowercase()
            .replace(' ', "_")
            .replace('\'', "")
            .replace(':', "");
        let base_dir = output_root.join(&folder_name);
        let room_names = room_names.into_iter().map(|(k, v)| (k as u16, v)).collect();
        Ok(Self {
            base_dir,
            room_names,
        })
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get the sanitized room folder name from RNAM or fallback to room_NNN
    pub fn room_folder_name(&self, room_id: u16) -> String {
        if let Some(name) = self.room_names.get(&room_id) {
            sanitize_name(name)
        } else {
            format!("room_{:03}", room_id)
        }
    }

    pub fn room_dir(&self, room_id: u16) -> PathBuf {
        self.base_dir
            .join("assets")
            .join("rooms")
            .join(self.room_folder_name(room_id))
    }

    pub fn room_image_dir(&self, room_id: u16) -> PathBuf {
        self.room_dir(room_id).join("images")
    }

    pub fn room_object_dir(&self, room_id: u16, obj_name: &str) -> PathBuf {
        self.room_image_dir(room_id)
            .join("objects")
            .join(sanitize_name(obj_name))
    }

    pub fn room_audio_dir(&self, room_id: u16) -> PathBuf {
        self.room_dir(room_id).join("audio")
    }

    pub fn room_sprite_dir(&self, room_id: u16) -> PathBuf {
        self.room_dir(room_id).join("sprites")
    }

    pub fn speech_dir(&self) -> PathBuf {
        self.base_dir.join("assets").join("speech")
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(self.base_dir.join("assets").join("rooms"))?;
        Ok(())
    }

    pub fn ensure_room_dirs(&self, room_id: u16) -> Result<()> {
        std::fs::create_dir_all(self.room_image_dir(room_id))?;
        Ok(())
    }

    pub fn ensure_room_object_dir(&self, room_id: u16, obj_name: &str) -> Result<()> {
        std::fs::create_dir_all(self.room_object_dir(room_id, obj_name))?;
        Ok(())
    }

    pub fn ensure_audio_room_dir(&self, room_id: u16) -> Result<()> {
        std::fs::create_dir_all(self.room_audio_dir(room_id))?;
        Ok(())
    }

    pub fn ensure_speech_dir(&self) -> Result<()> {
        std::fs::create_dir_all(self.speech_dir())?;
        Ok(())
    }

    pub fn ensure_sprite_dir(&self, room_id: u16) -> Result<()> {
        std::fs::create_dir_all(self.room_sprite_dir(room_id))?;
        Ok(())
    }

    pub fn lora_dir(&self) -> PathBuf {
        self.base_dir.join("lora_training")
    }

    pub fn lora_backgrounds_dir(&self) -> PathBuf {
        self.lora_dir().join("backgrounds")
    }

    pub fn lora_objects_dir(&self) -> PathBuf {
        self.lora_dir().join("objects")
    }

    pub fn lora_sprites_dir(&self) -> PathBuf {
        self.lora_dir().join("sprites")
    }

    pub fn lora_backgrounds_images_dir(&self) -> PathBuf {
        self.lora_backgrounds_dir().join("images")
    }

    pub fn lora_objects_images_dir(&self) -> PathBuf {
        self.lora_objects_dir().join("images")
    }

    pub fn lora_sprites_images_dir(&self) -> PathBuf {
        self.lora_sprites_dir().join("images")
    }

    pub fn ensure_lora_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(self.lora_backgrounds_images_dir())?;
        std::fs::create_dir_all(self.lora_objects_images_dir())?;
        std::fs::create_dir_all(self.lora_sprites_images_dir())?;
        Ok(())
    }
}

pub fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "_")
        .replace('\'', "")
        .replace(':', "")
        .replace('/', "_")
        .replace('\\', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect()
}

/// Trait for images that can be saved as PNG via palette lookup.
pub trait PaletteImage {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn pixels(&self) -> &[u8];
    fn palette_color(&self, index: u8) -> (u8, u8, u8);
}

/// Save a palette-indexed image as an RGB PNG.
pub fn save_image(img: &dyn PaletteImage, path: &Path) -> Result<()> {
    let mut rgb_data = Vec::with_capacity(img.pixels().len() * 3);
    for &idx in img.pixels() {
        let (r, g, b) = img.palette_color(idx);
        rgb_data.push(r);
        rgb_data.push(g);
        rgb_data.push(b);
    }

    let rgb_image = RgbImage::from_raw(img.width() as u32, img.height() as u32, rgb_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    rgb_image
        .save(path)
        .with_context(|| format!("Failed to save {}", path.display()))?;
    Ok(())
}

pub fn write_jsonl(entries: &[LoraEntry], path: &Path) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for entry in entries {
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{}", line)?;
    }
    Ok(())
}
