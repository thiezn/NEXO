use sqlx::SqlitePool;

/// Return whether a run has already been cancelled.
pub async fn run_cancelled(pool: &SqlitePool, run_id: &str) -> bool {
    match crate::agent::session::is_run_cancelled(pool, run_id).await {
        Ok(cancelled) => cancelled,
        Err(error) => {
            tracing::error!("Failed to load run status for {run_id}: {error}");
            false
        }
    }
}
