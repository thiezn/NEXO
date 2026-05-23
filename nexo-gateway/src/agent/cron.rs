use nexo_ws_schema::{CronEntry, CronPayload, EventKind, Frame};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

/// Create a new cron job. Returns the job ID.
pub async fn create_job(
    pool: &SqlitePool,
    name: &str,
    schedule: &str,
    prompt: &str,
    session_id: Option<&str>,
) -> Result<String, sqlx::Error> {
    let id = Frame::new_id();
    sqlx::query(
        "INSERT INTO cron_jobs (id, name, schedule, prompt, session_id) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(name)
    .bind(schedule)
    .bind(prompt)
    .bind(session_id)
    .execute(pool)
    .await?;
    Ok(id)
}

/// List all cron jobs.
pub async fn list_jobs(pool: &SqlitePool) -> Result<Vec<CronEntry>, sqlx::Error> {
    let rows: Vec<(String, String, String, bool, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT id, name, schedule, enabled, last_run_at, next_run_at
            FROM cron_jobs ORDER BY created_at",
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(id, name, schedule, enabled, last_run_at, next_run_at)| CronEntry {
                job_id: id,
                name,
                schedule,
                enabled,
                last_run_at,
                next_run_at,
            },
        )
        .collect())
}

/// Delete a cron job. Returns true if the job existed.
pub async fn delete_job(pool: &SqlitePool, job_id: &str) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM cron_jobs WHERE id = ?")
        .bind(job_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Background scheduler task. Checks every 60s for due jobs and fires them.
pub async fn run_scheduler(
    pool: SqlitePool,
    agent_handle: super::AgentHandle,
    event_tx: broadcast::Sender<Frame>,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
    loop {
        interval.tick().await;

        // Periodically reap expired capability locks
        let _ = super::locks::reap_expired(&pool).await;

        let due_jobs: Result<Vec<(String, String, String, Option<String>)>, _> = sqlx::query_as(
            "SELECT id, name, prompt, session_id FROM cron_jobs
             WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= datetime('now')",
        )
        .fetch_all(&pool)
        .await;

        let jobs = match due_jobs {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("Cron scheduler query failed: {e}");
                continue;
            }
        };

        for (job_id, name, prompt, session_id) in jobs {
            tracing::info!("Cron job firing: {name} ({job_id})");

            // Emit cron event
            let payload = CronPayload {
                job_id: job_id.clone(),
                name: name.clone(),
            };
            if let Ok(frame) = Frame::event(EventKind::Cron, &payload) {
                let _ = event_tx.send(frame);
            }

            // Always update last_run_at to prevent retry storms on persistent failures
            let update_timestamps = || async {
                let _ = sqlx::query(
                    "UPDATE cron_jobs SET last_run_at = datetime('now'), next_run_at = NULL WHERE id = ?",
                )
                .bind(&job_id)
                .execute(&pool)
                .await;
            };

            // Resolve or create session for the cron job
            let run_session_id = match session_id {
                Some(sid) => sid,
                None => {
                    match super::session::create_session(&pool, "cron", Some(&name), None).await {
                        Ok((sid, _)) => sid,
                        Err(e) => {
                            tracing::warn!("Failed to create cron session: {e}");
                            update_timestamps().await;
                            continue;
                        }
                    }
                }
            };

            let run_id = Frame::new_id();
            let idem_key = format!("cron-{job_id}-{}", chrono::Utc::now().timestamp());

            if let Err(e) =
                super::session::create_run(&pool, &run_id, &run_session_id, &idem_key, None, false)
                    .await
            {
                tracing::warn!("Failed to create cron run: {e}");
                update_timestamps().await;
                continue;
            }

            let cmd = super::AgentCommand::RunAgent {
                run_id,
                session_id: run_session_id,
                prompt,
                context: None,
                peer_id: "cron".into(),
                model_id: None,
                prefill_collection_id: None,
                thinking: false,
            };
            if let Err(e) = agent_handle.submit(cmd).await {
                tracing::warn!("Failed to submit cron agent command: {e}");
            }

            update_timestamps().await;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn create_and_list_jobs(pool: SqlitePool) {
        let id1 = create_job(&pool, "daily", "0 9 * * *", "summarize", None)
            .await
            .unwrap();
        let id2 = create_job(&pool, "hourly", "0 * * * *", "check status", None)
            .await
            .unwrap();
        assert_ne!(id1, id2);

        let jobs = list_jobs(&pool).await.unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].name, "daily");
        assert_eq!(jobs[1].name, "hourly");
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn delete_job_removes_it(pool: SqlitePool) {
        let id = create_job(&pool, "test", "* * * * *", "hello", None)
            .await
            .unwrap();
        let deleted = delete_job(&pool, &id).await.unwrap();
        assert!(deleted);

        let jobs = list_jobs(&pool).await.unwrap();
        assert!(jobs.is_empty());
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn delete_nonexistent_returns_false(pool: SqlitePool) {
        let deleted = delete_job(&pool, "nonexistent").await.unwrap();
        assert!(!deleted);
    }
}
