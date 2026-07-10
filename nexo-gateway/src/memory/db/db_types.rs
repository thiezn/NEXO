//! These types represent the internal database row projections and serde helpers for persistence.
//! They are not intended to be used outside of the database module, so its not allowed to have them as inputs
//! or outputs of the public DbClient API methods.
//! 
//! Types that generalize to application-level concepts (like PeerId, OperationId, etc.) are defined in nexo-core
//! and should be used instead of duplicating them here.

use crate::Result;
use crate::agent::{AgentJobKind, AgentJobQueueStatus, InferenceRunStateKind};
use nexo_core::{
    ModelId, OperationId, PeerId,
};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::SqliteRow;
use uuid::Uuid;

/// Internal row projection of an inference run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunRecord {
    /// Stable operation identifier.
    pub operation_id: OperationId,
    /// Current persisted run state.
    pub run_state: InferenceRunStateKind,
    /// Selected node, if known.
    pub node_peer_id: Option<PeerId>,
    /// Selected model, if known.
    pub model_id: Option<ModelId>,
    /// Persisted failure message, if any.
    pub error_message: Option<String>,
}

/// Internal row projection of a queued inference job.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentJobQueueRecord {
    /// FIFO ordering column.
    pub queue_position: i64,
    /// Stable operation identifier.
    pub operation_id: OperationId,
    /// User peer that owns the operation.
    pub user_peer_id: PeerId,
    /// Category of queued job.
    pub job_kind: AgentJobKind,
    /// Queue lifecycle status.
    pub status: AgentJobQueueStatus,
    /// Total claim attempts so far.
    pub attempt_count: i64,
    /// Failure message for failed jobs.
    pub failure_message: Option<String>,
}

impl<'r> sqlx::FromRow<'r, SqliteRow> for InferenceRunRecord {
    fn from_row(row: &SqliteRow) -> std::result::Result<Self, sqlx::Error> {
        let node_client_id: Option<String> = row.try_get("node_client_id")?;
        let node_device_id: Option<String> = row.try_get("node_device_id")?;

        Ok(Self {
            operation_id: row.try_get("operation_id")?,
            run_state: parse_enum_value("run_state", row.try_get::<String, _>("run_state")?)?,
            node_peer_id: optional_peer_id_from_parts(node_client_id, node_device_id)?,
            model_id: row.try_get("model_id")?,
            error_message: row.try_get("error_message")?,
        })
    }
}

impl<'r> sqlx::FromRow<'r, SqliteRow> for AgentJobQueueRecord {
    fn from_row(row: &SqliteRow) -> std::result::Result<Self, sqlx::Error> {
        Ok(Self {
            queue_position: row.try_get("queue_position")?,
            operation_id: row.try_get("operation_id")?,
            user_peer_id: peer_id_from_columns(row, "user_client_id", "user_device_id")?,
            job_kind: parse_enum_value("job_kind", row.try_get::<String, _>("job_kind")?)?,
            status: parse_enum_value("status", row.try_get::<String, _>("status")?)?,
            attempt_count: row.try_get("attempt_count")?,
            failure_message: row.try_get("failure_message")?,
        })
    }
}

/// Serialize a JSON payload for persistence.
///
/// # Arguments
///
/// * `value` - The serializable value to encode as a JSON string.
pub fn to_json_string<T: Serialize>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

/// Deserialize a JSON payload loaded from persistence.
///
/// # Arguments
///
/// * `value` - The JSON string loaded from the database.
pub fn from_json_str<T: for<'de> Deserialize<'de>>(value: &str) -> Result<T> {
    Ok(serde_json::from_str(value)?)
}

fn peer_id_from_columns(
    row: &SqliteRow,
    client_column: &'static str,
    device_column: &'static str,
) -> std::result::Result<PeerId, sqlx::Error> {
    let client_id: String = row.try_get(client_column)?;
    let device_id: String = row.try_get(device_column)?;
    peer_id_from_parts(client_id, device_id)
}

fn optional_peer_id_from_parts(
    client_id: Option<String>,
    device_id: Option<String>,
) -> std::result::Result<Option<PeerId>, sqlx::Error> {
    match (client_id, device_id) {
        (Some(client_id), Some(device_id)) => Ok(Some(peer_id_from_parts(client_id, device_id)?)),
        (None, None) => Ok(None),
        _ => Err(decode_error(
            "peer_id",
            "partial composite peer key; both client_id and device_id are required",
        )),
    }
}

pub(super) fn peer_id_from_parts(
    client_id: String,
    device_id: String,
) -> std::result::Result<PeerId, sqlx::Error> {
    let client_id = Uuid::parse_str(&client_id)
        .map_err(|error| decode_error("client_id", &format!("{client_id}: {error}")))?;
    let device_id = Uuid::parse_str(&device_id)
        .map_err(|error| decode_error("device_id", &format!("{device_id}: {error}")))?;
    Ok(PeerId::new(client_id, device_id))
}

fn parse_enum_value<T>(field: &'static str, value: String) -> std::result::Result<T, sqlx::Error>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| decode_error(field, &format!("{value}: {error}")))
}

fn decode_error(field: &'static str, value: &str) -> sqlx::Error {
    sqlx::Error::Decode(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid {field} value {value}"),
    )))
}
