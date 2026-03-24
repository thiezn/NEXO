use nexo_ws_schema::{EventKind, Frame, TickPayload};
use tokio::sync::broadcast;
use tokio::time::{Duration, interval};

/// Run the tick event broadcaster. Sends periodic `tick` events on the broadcast channel.
pub async fn run_ticker(tx: broadcast::Sender<Frame>, interval_ms: u64) {
    let mut ticker = interval(Duration::from_millis(interval_ms));
    let mut seq: u64 = 0;

    loop {
        ticker.tick().await;
        seq += 1;

        let payload = TickPayload {
            timestamp: chrono::Utc::now().to_rfc3339(),
            seq,
        };

        match Frame::event_with_seq(EventKind::Tick, &payload, seq) {
            Ok(frame) => {
                // Ignore send errors (no receivers connected)
                let _ = tx.send(frame);
            }
            Err(e) => {
                tracing::warn!("Failed to serialize tick event: {e}");
            }
        }
    }
}
