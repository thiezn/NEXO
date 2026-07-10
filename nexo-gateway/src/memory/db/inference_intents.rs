use super::DbClient;
use crate::{Error, Result};
use nexo_core::{InferenceIntent, InferenceOperationKind, OperationId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredInferenceIntent {
    operation_id: nexo_core::OperationId,
    session_id: nexo_core::SessionId,
    model_selection: nexo_core::ModelSelection,
    operation: nexo_core::InferenceOperation,
}

impl From<&InferenceIntent> for StoredInferenceIntent {
    fn from(value: &InferenceIntent) -> Self {
        Self {
            operation_id: value.operation_id,
            session_id: value.session_id,
            model_selection: value.model_selection.clone(),
            operation: value.operation.clone(),
        }
    }
}

impl From<StoredInferenceIntent> for InferenceIntent {
    fn from(value: StoredInferenceIntent) -> Self {
        Self {
            operation_id: value.operation_id,
            session_id: value.session_id,
            model_selection: value.model_selection,
            operation: value.operation,
        }
    }
}

impl DbClient {
    /// Persist a canonical inference intent for a known operation.
    ///
    /// # Arguments
    ///
    /// * `intent` - The full inference intent payload to serialize and store.
    pub async fn upsert_inference_intent(&self, intent: &InferenceIntent) -> Result {
        let now = Self::current_timestamp();
        sqlx::query(
            "INSERT INTO inference_intents (operation_id, session_id, operation_kind, model_selection_json, intent_json, created_at, updated_at)\n             VALUES (?, ?, ?, ?, ?, ?, ?)\n             ON CONFLICT(operation_id) DO UPDATE SET\n                session_id = excluded.session_id,\n                operation_kind = excluded.operation_kind,\n                model_selection_json = excluded.model_selection_json,\n                intent_json = excluded.intent_json,\n                updated_at = excluded.updated_at",
        )
        .bind(intent.operation_id.to_string())
        .bind(intent.session_id.to_string())
        .bind(InferenceOperationKind::from(&intent.operation).to_string())
        .bind(serde_json::to_string(&intent.model_selection)?)
        .bind(serde_json::to_string(&StoredInferenceIntent::from(intent))?)
        .bind(&now)
        .bind(&now)
        .execute(self.pool())
        .await?;

        Ok(())
    }

    /// Load a canonical inference intent by operation identifier.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation identifier whose stored intent should be loaded.
    pub async fn get_inference_intent(&self, operation_id: OperationId) -> Result<InferenceIntent> {
        let row = sqlx::query_as::<_, (String,)>(
            "SELECT intent_json FROM inference_intents WHERE operation_id = ?",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        let Some((intent_json,)) = row else {
            return Err(Error::NotFound {
                resource: "inference_intent",
                identifier: operation_id.to_string(),
            });
        };

        Ok(InferenceIntent::from(serde_json::from_str::<StoredInferenceIntent>(&intent_json)?))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use nexo_core::{
        ClientInfo, ConversationMessage, DeviceInfo, InferenceOperation, ModelCapability,
        ModelSelection, OperationId, ReasoningSettings, SessionId, ToolChoice, User, UserProperties,
    };
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_db() -> DbClient {
        let pool = SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let db = DbClient::from_pool(pool);
        db.initialize_schema().await.unwrap();
        db
    }

    fn test_user() -> User {
        let properties = UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_intent() -> InferenceIntent {
        InferenceIntent {
            operation_id: OperationId::new(),
            session_id: SessionId::new(),
            model_selection: ModelSelection::Capabilities(vec![ModelCapability::TextGeneration]),
            operation: InferenceOperation::MultiModal(MultiModalPayload::new_round(
                vec![ConversationMessage::new_text("hello")],
                Vec::new(),
                ToolChoice::Automatic,
                ReasoningSettings::default(),
            )),
        }
    }

    #[tokio::test]
    async fn upsert_and_get_inference_intent_round_trips_json_payload() {
        let db = test_db().await;
        let user = test_user();
        let intent = test_intent();

        db.connect_user(&user).await.unwrap();
        db.create_operation(intent.operation_id, user.id()).await.unwrap();
        db.upsert_inference_intent(&intent).await.unwrap();

        let stored = db.get_inference_intent(intent.operation_id).await.unwrap();
        assert_eq!(stored, intent);
        assert_eq!(stored.operation_kind(), intent.operation_kind());
    }
}
