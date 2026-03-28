use sqlx::SqlitePool;

/// Remove expired locks. Called periodically rather than on every acquire.
pub async fn reap_expired(pool: &SqlitePool) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM capability_locks WHERE expires_at < datetime('now')")
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Try to acquire an exclusive lock on a capability for the given run.
/// Returns true if acquired, false if already locked by another run.
/// Uses INSERT OR IGNORE so an existing (even expired) row blocks; call `reap_expired` periodically.
pub async fn acquire(
    pool: &SqlitePool,
    capability: &str,
    run_id: &str,
) -> Result<bool, sqlx::Error> {
    // Attempt atomic insert (PRIMARY KEY conflict = already locked)
    let result = sqlx::query(
        "INSERT OR IGNORE INTO capability_locks (capability, run_id, expires_at)
         VALUES (?, ?, datetime('now', '+5 minutes'))",
    )
    .bind(capability)
    .bind(run_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() == 1)
}

/// Release the lock on a capability.
pub async fn release(pool: &SqlitePool, capability: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM capability_locks WHERE capability = ?")
        .bind(capability)
        .execute(pool)
        .await?;
    Ok(())
}

/// Release all locks held by a specific run (cleanup on failure or completion).
pub async fn release_all_for_run(pool: &SqlitePool, run_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM capability_locks WHERE run_id = ?")
        .bind(run_id)
        .execute(pool)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Helper to set up the required FK parent rows for capability_locks.
    async fn setup_run(pool: &SqlitePool, run_id: &str) {
        sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
            .execute(pool)
            .await
            .ok(); // ignore if already exists
        sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
            .execute(pool)
            .await
            .ok();
        sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('s1', 'u1')")
            .execute(pool)
            .await
            .ok();
        sqlx::query("INSERT INTO agent_runs (id, session_id, idempotency_key) VALUES (?, 's1', ?)")
            .bind(run_id)
            .bind(format!("idem-{run_id}"))
            .execute(pool)
            .await
            .ok();
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn acquire_lock_succeeds_when_free(pool: SqlitePool) {
        setup_run(&pool, "run-1").await;
        let acquired = acquire(&pool, "llm", "run-1").await.unwrap();
        assert!(acquired);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn acquire_lock_fails_when_held(pool: SqlitePool) {
        setup_run(&pool, "run-1").await;
        setup_run(&pool, "run-2").await;

        let first = acquire(&pool, "llm", "run-1").await.unwrap();
        assert!(first);

        let second = acquire(&pool, "llm", "run-2").await.unwrap();
        assert!(!second);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn acquire_lock_succeeds_after_release(pool: SqlitePool) {
        setup_run(&pool, "run-1").await;
        setup_run(&pool, "run-2").await;

        acquire(&pool, "llm", "run-1").await.unwrap();
        release(&pool, "llm").await.unwrap();

        let acquired = acquire(&pool, "llm", "run-2").await.unwrap();
        assert!(acquired);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn expired_lock_is_reaped(pool: SqlitePool) {
        setup_run(&pool, "run-1").await;
        setup_run(&pool, "run-2").await;

        // Insert a lock with an already-expired timestamp
        sqlx::query(
            "INSERT INTO capability_locks (capability, run_id, expires_at)
             VALUES ('llm', 'run-1', datetime('now', '-1 minute'))",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Reap expired locks, then acquire should succeed
        let reaped = reap_expired(&pool).await.unwrap();
        assert_eq!(reaped, 1);

        let acquired = acquire(&pool, "llm", "run-2").await.unwrap();
        assert!(acquired);
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn release_all_for_run_clears_locks(pool: SqlitePool) {
        setup_run(&pool, "run-1").await;

        acquire(&pool, "llm", "run-1").await.unwrap();
        acquire(&pool, "tts", "run-1").await.unwrap();

        release_all_for_run(&pool, "run-1").await.unwrap();

        let (count,): (i32,) =
            sqlx::query_as("SELECT COUNT(*) FROM capability_locks WHERE run_id = 'run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 0);
    }
}
