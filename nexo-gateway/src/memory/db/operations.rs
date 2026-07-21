use super::DbClient;
#[cfg(test)]
use crate::Error;
use crate::Result;
use nexo_core::{OperationId, PeerId};

impl DbClient {
    /// Create or refresh the owner mapping for an operation.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The stable operation identifier to persist.
    /// * `user_peer_id` - The requesting user peer that owns the operation.
    pub async fn create_operation(
        &self,
        operation_id: OperationId,
        user_peer_id: PeerId,
    ) -> Result {
        let created_at = Self::current_timestamp();
        sqlx::query(
            "INSERT INTO operations (operation_id, user_client_id, user_device_id, created_at) VALUES (?, ?, ?, ?)\n             ON CONFLICT(operation_id) DO UPDATE SET\n                user_client_id = excluded.user_client_id,\n                user_device_id = excluded.user_device_id",
        )
        .bind(operation_id.to_string())
        .bind(user_peer_id.client_id().to_string())
        .bind(user_peer_id.device_id().to_string())
        .bind(created_at)
        .execute(self.pool())
        .await?;

        Ok(())
    }

    /// Load the persisted owner mapping for an operation.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The stable operation identifier to load.
    #[cfg(test)]
    async fn get_operation(&self, operation_id: OperationId) -> Result<(OperationId, PeerId)> {
        let row = sqlx::query_as::<_, (OperationId, String, String)>(
            "SELECT operation_id, user_client_id, user_device_id FROM operations WHERE operation_id = ?",
        )
        .bind(operation_id.to_string())
        .fetch_optional(self.pool())
        .await?;

        let Some((operation_id, user_client_id, user_device_id)) = row else {
            return Err(Error::NotFound {
                resource: "operation",
                identifier: operation_id.to_string(),
            });
        };

        let user_peer_id = super::db_types::peer_id_from_parts(user_client_id, user_device_id)?;
        Ok((operation_id, user_peer_id))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, User, UserProperties};
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
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    #[tokio::test]
    async fn create_and_get_operation_round_trips_owner() {
        let db = test_db().await;
        let user = test_user();
        db.connect_user(&user).await.unwrap();

        let operation_id = OperationId::new();
        db.create_operation(operation_id, user.id()).await.unwrap();

        let (stored_operation_id, stored_user_peer_id) =
            db.get_operation(operation_id).await.unwrap();
        assert_eq!(stored_operation_id, operation_id);
        assert_eq!(stored_user_peer_id, user.id());
    }
}
