//! These types represent the internal database row projections and serde helpers for persistence.
//! They are not intended to be used outside of the database module, so its not allowed to have them as inputs
//! or outputs of the public DbClient API methods.
//! 
//! Types that generalize to application-level concepts (like PeerId, OperationId, etc.) are defined in nexo-core
//! and should be used instead of duplicating them here.

use crate::Error;
use crate::Result;
use crate::agent::{
    AgentJobKind, AgentJobQueueStatus, InferenceRunSnapshot, InferenceRunState,
    InferenceRunStateKind, InferenceRunTimeline,
};
use nexo_core::{ModelId, OperationId, PeerId};
use sqlx::Row;
use sqlx::sqlite::SqliteRow;
use uuid::Uuid;

/// Internal row projection of the `inference_runs` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunRow {
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
    /// When the run row was first created.
    pub created_at: String,
    /// When context preparation began.
    pub preparing_started_at: Option<String>,
    /// When a node/model selection was first persisted.
    pub node_selected_at: Option<String>,
    /// When model loading began.
    pub model_loading_started_at: Option<String>,
    /// When inference execution began.
    pub in_progress_at: Option<String>,
    /// When the run completed.
    pub completed_at: Option<String>,
    /// When the run failed.
    pub failed_at: Option<String>,
    /// When the persisted state last changed.
    pub last_state_changed_at: String,
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

impl<'r> sqlx::FromRow<'r, SqliteRow> for InferenceRunRow {
    fn from_row(row: &SqliteRow) -> std::result::Result<Self, sqlx::Error> {
        let node_client_id: Option<String> = row.try_get("node_client_id")?;
        let node_device_id: Option<String> = row.try_get("node_device_id")?;

        Ok(Self {
            operation_id: row.try_get("operation_id")?,
            run_state: parse_enum_value("run_state", row.try_get::<String, _>("run_state")?)?,
            node_peer_id: optional_peer_id_from_parts(node_client_id, node_device_id)?,
            model_id: row.try_get("model_id")?,
            error_message: row.try_get("error_message")?,
            created_at: row.try_get("created_at")?,
            preparing_started_at: row.try_get("preparing_started_at")?,
            node_selected_at: row.try_get("node_selected_at")?,
            model_loading_started_at: row.try_get("model_loading_started_at")?,
            in_progress_at: row.try_get("in_progress_at")?,
            completed_at: row.try_get("completed_at")?,
            failed_at: row.try_get("failed_at")?,
            last_state_changed_at: row.try_get("last_state_changed_at")?,
        })
    }
}

impl TryFrom<InferenceRunRow> for InferenceRunSnapshot {
    type Error = Error;

    fn try_from(value: InferenceRunRow) -> Result<Self, Self::Error> {
        let state = match value.run_state {
            InferenceRunStateKind::Queued => InferenceRunState::Queued,
            InferenceRunStateKind::PreparingContext => InferenceRunState::PreparingContext,
            InferenceRunStateKind::UnloadingModel => InferenceRunState::UnloadingModel {
                node_peer_id: required_field(value.node_peer_id, "node_peer_id", value.run_state)?,
                model_id: required_field(value.model_id, "model_id", value.run_state)?,
            },
            InferenceRunStateKind::LoadingModel => InferenceRunState::LoadingModel {
                node_peer_id: required_field(value.node_peer_id, "node_peer_id", value.run_state)?,
                model_id: required_field(value.model_id, "model_id", value.run_state)?,
            },
            InferenceRunStateKind::InProgress => InferenceRunState::InProgress {
                node_peer_id: required_field(value.node_peer_id, "node_peer_id", value.run_state)?,
                model_id: required_field(value.model_id, "model_id", value.run_state)?,
            },
            InferenceRunStateKind::Completed => InferenceRunState::Completed {
                node_peer_id: required_field(value.node_peer_id, "node_peer_id", value.run_state)?,
                model_id: required_field(value.model_id, "model_id", value.run_state)?,
            },
            InferenceRunStateKind::Failed => InferenceRunState::Failed {
                error_message: required_field(
                    value.error_message.clone(),
                    "error_message",
                    value.run_state,
                )?,
                node_peer_id: value.node_peer_id,
                model_id: value.model_id,
            },
        };

        Ok(InferenceRunSnapshot {
            operation_id: value.operation_id,
            state,
            timeline: InferenceRunTimeline {
                created_at: value.created_at,
                preparing_started_at: value.preparing_started_at,
                node_selected_at: value.node_selected_at,
                model_loading_started_at: value.model_loading_started_at,
                in_progress_at: value.in_progress_at,
                completed_at: value.completed_at,
                failed_at: value.failed_at,
                last_state_changed_at: value.last_state_changed_at,
            },
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

/// Require a persisted field to exist for a specific run state.
///
/// # Arguments
///
/// * `value` - The optional persisted value to validate.
/// * `field` - The logical field name used in the validation error.
/// * `run_state` - The persisted run-state discriminant being reconstructed.
fn required_field<T>(
    value: Option<T>,
    field: &'static str,
    run_state: InferenceRunStateKind,
) -> Result<T, Error> {
    value.ok_or_else(|| {
        Error::InvalidInferenceRunState(format!(
            "missing {field} for persisted state {run_state}"
        ))
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, User, UserProperties};

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_row(run_state: InferenceRunStateKind) -> InferenceRunRow {
        InferenceRunRow {
            operation_id: OperationId::new(),
            run_state,
            node_peer_id: None,
            model_id: None,
            error_message: None,
            created_at: "2026-07-10T10:00:00Z".into(),
            preparing_started_at: None,
            node_selected_at: None,
            model_loading_started_at: None,
            in_progress_at: None,
            completed_at: None,
            failed_at: None,
            last_state_changed_at: "2026-07-10T10:00:01Z".into(),
        }
    }

    #[test]
    fn loading_row_requires_selected_node_and_model() {
        let error = InferenceRunSnapshot::try_from(test_row(InferenceRunStateKind::LoadingModel))
            .expect_err("expected invalid persisted loading_model row");

        assert!(matches!(error, Error::InvalidInferenceRunState(_)));
    }

    #[test]
    fn failed_row_requires_error_message() {
        let error = InferenceRunSnapshot::try_from(test_row(InferenceRunStateKind::Failed))
            .expect_err("expected invalid persisted failed row");

        assert!(matches!(error, Error::InvalidInferenceRunState(_)));
    }

    #[test]
    fn unloading_row_reconstructs_snapshot() {
        let user = test_user();
        let mut row = test_row(InferenceRunStateKind::UnloadingModel);
        row.node_peer_id = Some(user.id());
        row.model_id = Some(ModelId::Kokoro82m);
        row.preparing_started_at = Some("2026-07-10T10:00:01Z".into());
        row.node_selected_at = Some("2026-07-10T10:00:02Z".into());
        row.last_state_changed_at = "2026-07-10T10:00:02Z".into();

        let snapshot = InferenceRunSnapshot::try_from(row).unwrap();

        let InferenceRunState::UnloadingModel {
            node_peer_id,
            model_id,
        } = snapshot.state
        else {
            panic!("expected unloading_model snapshot")
        };

        assert_eq!(node_peer_id, user.id());
        assert_eq!(model_id, ModelId::Kokoro82m);
        assert_eq!(snapshot.timeline.node_selected_at.as_deref(), Some("2026-07-10T10:00:02Z"));
    }
}
