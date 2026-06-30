use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use std::path::Path;

/// Initialize the SQLite database at the given path.
pub async fn initialize(db_path: &Path) -> cli_helpers::Result {
    let pool = connect(db_path).await?;
    pool.close().await;

    tracing::info!("Database initialized at {}", db_path.display());
    Ok(())
}

/// Connect to the SQLite database at the given path, ensuring schema migrations are applied.
pub async fn connect(db_path: &Path) -> cli_helpers::Result<SqlitePool> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            cli_helpers::Error::Other(format!(
                "Failed to create DB directory '{}': {e}",
                parent.display()
            ))
        })?;
    }

    let url = format!("sqlite:{}?mode=rwc", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .map_err(|e| cli_helpers::Error::Other(format!("Failed to connect to DB: {e}")))?;

    run_migrations(&pool).await?;
    Ok(pool)
}

async fn run_migrations(pool: &SqlitePool) -> cli_helpers::Result {
    sqlx::migrate!("./migrations")
        .run(pool)
        .await
        .map_err(|e| cli_helpers::Error::Other(format!("Migration failed: {e}")))?;
    Ok(())
}

/// Record or update a device in the persistent store.
pub async fn upsert_device(
    pool: &SqlitePool,
    device_id: &str,
    role: nexo_core::NexoClientKind,
) -> cli_helpers::Result {
    sqlx::query(
        "INSERT INTO devices (id, role) VALUES (?, ?)
         ON CONFLICT(id) DO UPDATE SET role = excluded.role, last_seen = datetime('now')",
    )
    .bind(device_id)
    .bind(role.to_string())
    .execute(pool)
    .await
    .map_err(|e| cli_helpers::Error::Other(format!("Failed to upsert device: {e}")))?;
    Ok(())
}

/// Record or update a user in the persistent store.
pub async fn upsert_user(pool: &SqlitePool, user_id: &str, device_id: &str) -> cli_helpers::Result {
    sqlx::query(
        "INSERT INTO users (id, device_id) VALUES (?, ?)
         ON CONFLICT(id) DO UPDATE SET device_id = excluded.device_id, last_seen = datetime('now')",
    )
    .bind(user_id)
    .bind(device_id)
    .execute(pool)
    .await
    .map_err(|e| cli_helpers::Error::Other(format!("Failed to upsert user: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use nexo_core::NexoClientKind;

    fn temp_db_path(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("nexo_test_{}_{nanos}.db", name))
    }

    #[sqlx::test]
    async fn initialize_creates_tables(pool: SqlitePool) {
        run_migrations(&pool).await.unwrap();

        let tables: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .fetch_all(&pool)
                .await
                .unwrap();

        let table_names: Vec<&str> = tables.iter().map(|t| t.0.as_str()).collect();
        assert!(table_names.contains(&"devices"));
        assert!(table_names.contains(&"users"));
        assert!(table_names.contains(&"idempotency_keys"));
    }

    #[sqlx::test]
    async fn upsert_device_inserts_and_updates(pool: SqlitePool) {
        run_migrations(&pool).await.unwrap();

        upsert_device(&pool, "dev-1", NexoClientKind::Node)
            .await
            .unwrap();

        let (role,): (String,) = sqlx::query_as("SELECT role FROM devices WHERE id = 'dev-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(role, "node");

        // Update role
        upsert_device(&pool, "dev-1", NexoClientKind::User)
            .await
            .unwrap();
        let (role,): (String,) = sqlx::query_as("SELECT role FROM devices WHERE id = 'dev-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(role, "user");
    }

    #[sqlx::test]
    async fn upsert_user_inserts_and_updates(pool: SqlitePool) {
        run_migrations(&pool).await.unwrap();

        upsert_device(&pool, "dev-1", NexoClientKind::User)
            .await
            .unwrap();
        upsert_user(&pool, "user-1", "dev-1").await.unwrap();

        let (device_id,): (String,) =
            sqlx::query_as("SELECT device_id FROM users WHERE id = 'user-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(device_id, "dev-1");
    }

    #[tokio::test]
    async fn initialize_from_path() {
        let path = temp_db_path("init");
        initialize(&path).await.unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn connect_from_path_runs_migrations() {
        let path = temp_db_path("connect");
        let pool = connect(&path).await.unwrap();

        let tables: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .fetch_all(&pool)
                .await
                .unwrap();

        let table_names: Vec<&str> = tables.iter().map(|t| t.0.as_str()).collect();
        assert!(table_names.contains(&"devices"));
        assert!(table_names.contains(&"users"));
        assert!(table_names.contains(&"_sqlx_migrations"));

        pool.close().await;
        let _ = std::fs::remove_file(&path);
    }
}
