use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ImageRef {
    pub path: String,
    pub id: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}
