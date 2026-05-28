#![allow(clippy::unwrap_used)]

use super::*;
use sqlx::SqlitePool;

fn disabled_reasoning() -> nexo_core::ReasoningSettings {
    nexo_core::ReasoningSettings::default()
}

fn enabled_reasoning() -> nexo_core::ReasoningSettings {
    nexo_core::ReasoningSettings {
        thinking: nexo_core::ThinkingMode::Enabled,
        effort: None,
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn create_session_generates_uuid(pool: SqlitePool) {
    // Insert a user first (FK constraint)
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('user-1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "user-1", Some("test session"), None::<&str>)
        .await
        .unwrap();
    assert!(!sid.is_empty());

    let (name,): (Option<String>,) = sqlx::query_as("SELECT name FROM sessions WHERE id = ?")
        .bind(&sid)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(name.as_deref(), Some("test session"));
}

#[sqlx::test(migrations = "./migrations")]
async fn list_sessions_returns_by_user(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u2', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    create_session(&pool, "u1", Some("s1"), None).await.unwrap();
    create_session(&pool, "u1", Some("s2"), None).await.unwrap();
    create_session(&pool, "u2", Some("s3"), None).await.unwrap();

    let u1_sessions = list_sessions(&pool, "u1").await.unwrap();
    assert_eq!(u1_sessions.len(), 2);

    let u2_sessions = list_sessions(&pool, "u2").await.unwrap();
    assert_eq!(u2_sessions.len(), 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn list_sessions_includes_message_count(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    insert_message(&pool, &sid, None, "user", "hello", None, None)
        .await
        .unwrap();
    insert_message(&pool, &sid, None, "assistant", "hi", None, None)
        .await
        .unwrap();

    let sessions = list_sessions(&pool, "u1").await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].message_count, 2);
}

#[sqlx::test(migrations = "./migrations")]
async fn get_session_returns_messages(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", Some("chat"), None)
        .await
        .unwrap();
    insert_message(&pool, &sid, None, "user", "hello", None, None)
        .await
        .unwrap();
    insert_message(&pool, &sid, None, "assistant", "hi back", None, None)
        .await
        .unwrap();

    let resp = get_session(&pool, &sid).await.unwrap().unwrap();
    assert_eq!(resp.session_id, sid);
    assert_eq!(resp.name.as_deref(), Some("chat"));
    assert_eq!(resp.messages.len(), 2);
    assert_eq!(resp.messages[0].role, nexo_ws_schema::MessageRole::User);
    assert_eq!(
        resp.messages[1].role,
        nexo_ws_schema::MessageRole::Assistant
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn get_session_not_found(pool: SqlitePool) {
    let resp = get_session(&pool, "nonexistent").await.unwrap();
    assert!(resp.is_none());
}

#[sqlx::test(migrations = "./migrations")]
async fn clear_session_deletes_everything(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(&pool, "run-1", &sid, "idem-1", None, &disabled_reasoning())
        .await
        .unwrap();
    insert_message(&pool, &sid, Some("run-1"), "user", "hello", None, None)
        .await
        .unwrap();

    let cleared = clear_session(&pool, &sid).await.unwrap();
    assert!(cleared);

    let (msg_count,): (i32,) =
        sqlx::query_as("SELECT COUNT(*) FROM conversation_entries WHERE session_id = ?")
            .bind(&sid)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(msg_count, 0);

    let (run_count,): (i32,) = sqlx::query_as("SELECT COUNT(*) FROM runs WHERE session_id = ?")
        .bind(&sid)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(run_count, 0);

    let (sess_count,): (i32,) = sqlx::query_as("SELECT COUNT(*) FROM sessions WHERE id = ?")
        .bind(&sid)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(sess_count, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn clear_nonexistent_returns_false(pool: SqlitePool) {
    let cleared = clear_session(&pool, "nonexistent").await.unwrap();
    assert!(!cleared);
}

#[sqlx::test(migrations = "./migrations")]
async fn clear_session_preserves_other_sessions(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (target_session_id, _) = create_session(&pool, "u1", Some("target"), None)
        .await
        .unwrap();
    let (other_session_id, _) = create_session(&pool, "u1", Some("keep"), None)
        .await
        .unwrap();

    create_run(
        &pool,
        "run-target",
        &target_session_id,
        "idem-target",
        None,
        &disabled_reasoning(),
    )
    .await
    .unwrap();
    create_run(
        &pool,
        "run-keep",
        &other_session_id,
        "idem-keep",
        None,
        &disabled_reasoning(),
    )
    .await
    .unwrap();
    insert_message(
        &pool,
        &target_session_id,
        Some("run-target"),
        "user",
        "delete me",
        None,
        None,
    )
    .await
    .unwrap();
    insert_message(
        &pool,
        &other_session_id,
        Some("run-keep"),
        "user",
        "keep me",
        None,
        None,
    )
    .await
    .unwrap();

    assert!(clear_session(&pool, &target_session_id).await.unwrap());

    let kept_session = get_session(&pool, &other_session_id).await.unwrap();
    assert!(kept_session.is_some());

    let (remaining_runs,): (i32,) =
        sqlx::query_as("SELECT COUNT(*) FROM runs WHERE session_id = ?")
            .bind(&other_session_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(remaining_runs, 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn insert_message_updates_last_active(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();

    let (before,): (String,) = sqlx::query_as("SELECT last_active_at FROM sessions WHERE id = ?")
        .bind(&sid)
        .fetch_one(&pool)
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

    insert_message(&pool, &sid, None, "user", "hello", None, None)
        .await
        .unwrap();

    let (after,): (String,) = sqlx::query_as("SELECT last_active_at FROM sessions WHERE id = ?")
        .bind(&sid)
        .fetch_one(&pool)
        .await
        .unwrap();

    assert!(after >= before);
}

#[sqlx::test(migrations = "./migrations")]
async fn create_and_finish_run(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(&pool, "run-1", &sid, "idem-1", None, &disabled_reasoning())
        .await
        .unwrap();

    let (status, reasoning): (String, String) =
        sqlx::query_as("SELECT status, reasoning FROM runs WHERE id = 'run-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(status, "accepted");
    assert_eq!(
        crate::agent::persistence::decode_reasoning_json(&reasoning).unwrap(),
        disabled_reasoning()
    );

    finish_run(
        &pool,
        "run-1",
        nexo_ws_schema::RunStatus::Completed,
        Some("All done"),
    )
    .await
    .unwrap();

    let (status, finished): (String, Option<String>) =
        sqlx::query_as("SELECT status, finished_at FROM runs WHERE id = 'run-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(status, "completed");
    assert!(finished.is_some());

    let (summary_kind, summary_content): (String, String) =
        sqlx::query_as("SELECT kind, content FROM run_summaries WHERE run_id = 'run-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(summary_kind, "final_response");
    assert_eq!(summary_content, "All done");
}

#[sqlx::test(migrations = "./migrations")]
async fn stop_run_marks_run_cancelled(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(&pool, "run-1", &sid, "idem-1", None, &disabled_reasoning())
        .await
        .unwrap();

    let stopped_session = stop_run(&pool, "run-1").await.unwrap();
    assert_eq!(stopped_session.as_deref(), Some(sid.as_str()));
    assert!(is_run_cancelled(&pool, "run-1").await.unwrap());
}

#[sqlx::test(migrations = "./migrations")]
async fn append_run_context_persists_system_message(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(&pool, "run-1", &sid, "idem-1", None, &disabled_reasoning())
        .await
        .unwrap();

    let message_id = append_run_context(&pool, "run-1", &serde_json::json!({"hint": "use notes"}))
        .await
        .unwrap();
    assert!(message_id.is_some());

    let row: (String, String) = sqlx::query_as(
        "SELECT role, content FROM conversation_entries WHERE run_id = 'run-1' ORDER BY created_at DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0, "system");
    assert!(row.1.contains("use notes"));
}

#[sqlx::test(migrations = "./migrations")]
async fn round_trace_and_summary_records_persist(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(
        &pool,
        "run-1",
        &sid,
        "idem-1",
        Some("gemma"),
        &enabled_reasoning(),
    )
    .await
    .unwrap();
    let round_id = create_round(&pool, "run-1", 1, Some("gemma"))
        .await
        .unwrap();
    let trace_id = create_tool_trace(
        &pool,
        "run-1",
        &round_id,
        "call-1",
        "echo.run",
        &serde_json::json!({"input": "hello"}),
    )
    .await
    .unwrap();
    finish_tool_trace(&pool, &trace_id, true, Some("hello"), None)
        .await
        .unwrap();
    finish_round(
        &pool,
        &round_id,
        "completed",
        Some("Reasoning"),
        Some("node-1"),
    )
    .await
    .unwrap();
    store_run_summary(&pool, "run-1", Some(&round_id), "final_response", "hello")
        .await
        .unwrap();

    let (round_status, rationale, selected_peer_id): (String, Option<String>, Option<String>) =
        sqlx::query_as("SELECT status, rationale, selected_peer_id FROM run_rounds WHERE id = ?")
            .bind(&round_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(round_status, "completed");
    assert_eq!(rationale.as_deref(), Some("Reasoning"));
    assert_eq!(selected_peer_id.as_deref(), Some("node-1"));

    let (tool_status, output): (String, Option<String>) =
        sqlx::query_as("SELECT status, output FROM tool_traces WHERE id = ?")
            .bind(&trace_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(tool_status, "completed");
    assert_eq!(output.as_deref(), Some("hello"));
}

#[sqlx::test(migrations = "./migrations")]
async fn next_round_index_starts_after_existing_rounds(pool: SqlitePool) {
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('u1', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    let (sid, _) = create_session(&pool, "u1", None, None).await.unwrap();
    create_run(
        &pool,
        "run-1",
        &sid,
        "idem-1",
        Some("gemma"),
        &disabled_reasoning(),
    )
    .await
    .unwrap();

    assert_eq!(next_round_index(&pool, "run-1").await.unwrap(), 1);

    create_round(&pool, "run-1", 1, Some("gemma"))
        .await
        .unwrap();

    assert_eq!(next_round_index(&pool, "run-1").await.unwrap(), 2);
}
