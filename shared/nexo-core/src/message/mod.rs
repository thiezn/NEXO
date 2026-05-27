//! Conversation messages and multimodal message parts.

/// Message content parts such as text, media, and tool payloads.
pub mod content_part;
/// Conversation and conversation message containers.
pub mod conversation;
/// Multimodal media input types.
pub mod multimodal;
/// Message role definitions.
pub mod role;

pub use content_part::{ContentPart, TextPart};
pub use conversation::{Conversation, ConversationMessage};
pub use multimodal::{AudioInput, ImageInput, MediaSource, VideoInput};
pub use role::MessageRole;
