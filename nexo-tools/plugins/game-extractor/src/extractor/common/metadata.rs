use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct RoomMetadata {
    pub room_id: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room_name: Option<String>,
    pub width: u16,
    pub height: u16,
    pub palette_type: String,
    pub num_objects: usize,
    pub objects: Vec<ObjectMetadataJson>,
    pub num_sounds: usize,
    pub sounds: Vec<SoundRef>,
    pub num_sprites: usize,
    pub sprites: Vec<SpriteMetadataJson>,
}

#[derive(Serialize)]
pub struct SpriteMetadataJson {
    pub sprite_index: usize,
    pub num_frames: usize,
    pub num_animations: usize,
    pub animations: Vec<SpriteAnimRef>,
    pub directory: String,
}

#[derive(Serialize)]
pub struct SpriteAnimRef {
    pub name: String,
    pub file: String,
    pub num_frames: usize,
}

#[derive(Serialize)]
pub struct ObjectMetadataJson {
    pub obj_id: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    pub num_states: u8,
    pub image_files: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LoraEntry {
    pub image: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct SoundRef {
    pub sound_id: u16,
    pub files: Vec<String>,
}

#[derive(Serialize)]
pub struct AudioMetadata {
    pub game: String,
    pub total_room_sounds: usize,
    pub total_speech: usize,
    pub room_sounds: Vec<SoundMetadataJson>,
    pub speech: Vec<SpeechMetadataJson>,
}

#[derive(Serialize)]
pub struct SoundMetadataJson {
    pub sound_id: u16,
    pub room_id: u16,
    pub sound_type: String,
    pub format: String,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}

#[derive(Serialize)]
pub struct SpeechMetadataJson {
    pub index: usize,
    pub format: String,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}
