use nexo_ws_schema::{CronCreateParams, CronDeleteParams, Frame};
use sqlx::SqlitePool;

use super::base::{internal_error, ok_or_internal_error, parse_params};

pub(super) async fn handle_create(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let cron_params: CronCreateParams = match parse_params(request_id, params, "cron.create") {
        Ok(p) => p,
        Err(f) => return f,
    };

    match crate::agent::cron::create_job(
        db,
        &cron_params.name,
        &cron_params.schedule,
        &cron_params.prompt,
        cron_params.session_id.as_deref(),
    )
    .await
    {
        Ok(job_id) => {
            ok_or_internal_error(request_id, nexo_ws_schema::CronCreateResponse { job_id })
        }
        Err(e) => internal_error(request_id, format!("Failed to create cron job: {e}")),
    }
}

pub(super) async fn handle_list(request_id: &str, db: &SqlitePool) -> Frame {
    match crate::agent::cron::list_jobs(db).await {
        Ok(jobs) => ok_or_internal_error(request_id, nexo_ws_schema::CronListResponse { jobs }),
        Err(e) => internal_error(request_id, format!("Failed to list cron jobs: {e}")),
    }
}

pub(super) async fn handle_delete(
    request_id: &str,
    params: serde_json::Value,
    db: &SqlitePool,
) -> Frame {
    let del_params: CronDeleteParams = match parse_params(request_id, params, "cron.delete") {
        Ok(p) => p,
        Err(f) => return f,
    };

    match crate::agent::cron::delete_job(db, &del_params.job_id).await {
        Ok(deleted) => {
            ok_or_internal_error(request_id, nexo_ws_schema::CronDeleteResponse { deleted })
        }
        Err(e) => internal_error(request_id, format!("Failed to delete cron job: {e}")),
    }
}
