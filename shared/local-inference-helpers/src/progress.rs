use std::time::Duration;

/// Progress events emitted during model loading and inference.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// A named stage has started (e.g. "Loading T5 encoder (CPU)")
    StageStart { name: String },
    /// The most recent stage completed, with its elapsed time.
    StageDone { name: String, elapsed: Duration },
    /// Informational message (e.g. "Metal detected, using GPU")
    Info { message: String },
    /// A single iterative step completed (denoising, decoding, etc.).
    Step {
        step: usize,
        total: usize,
        elapsed: Duration,
    },
}

/// Callback type for receiving progress events.
pub type ProgressCallback = Box<dyn Fn(ProgressEvent) + Send + Sync>;

/// Wrapper around an optional progress callback with convenience methods.
#[derive(Default)]
pub struct ProgressReporter {
    callback: Option<ProgressCallback>,
}

impl ProgressReporter {
    pub fn emit(&self, event: ProgressEvent) {
        if let Some(cb) = &self.callback {
            cb(event);
        }
    }

    pub fn stage_start(&self, name: &str) {
        self.emit(ProgressEvent::StageStart {
            name: name.to_string(),
        });
    }

    pub fn stage_done(&self, name: &str, elapsed: Duration) {
        self.emit(ProgressEvent::StageDone {
            name: name.to_string(),
            elapsed,
        });
    }

    pub fn info(&self, message: &str) {
        self.emit(ProgressEvent::Info {
            message: message.to_string(),
        });
    }

    pub fn set_callback(&mut self, callback: ProgressCallback) {
        self.callback = Some(callback);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn capturing_callback() -> (ProgressCallback, Arc<Mutex<Vec<String>>>) {
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);
        let cb: ProgressCallback = Box::new(move |event: ProgressEvent| {
            log_clone.lock().unwrap().push(format!("{event:?}"));
        });
        (cb, log)
    }

    #[test]
    fn default_no_callback_no_panic() {
        let reporter = ProgressReporter::default();
        reporter.stage_start("Loading model");
        reporter.stage_done("Loading model", Duration::from_millis(42));
        reporter.info("hello");
        reporter.emit(ProgressEvent::Step {
            step: 1,
            total: 10,
            elapsed: Duration::from_millis(5),
        });
    }

    #[test]
    fn callback_receives_stage_start() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.stage_start("Encoding prompt");

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("StageStart"));
        assert!(entries[0].contains("Encoding prompt"));
    }

    #[test]
    fn callback_receives_step() {
        let mut reporter = ProgressReporter::default();
        let (cb, log) = capturing_callback();
        reporter.set_callback(cb);

        reporter.emit(ProgressEvent::Step {
            step: 3,
            total: 20,
            elapsed: Duration::from_millis(100),
        });

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].contains("Step"));
        assert!(entries[0].contains("step: 3"));
    }

    #[test]
    fn set_callback_replaces_previous() {
        let mut reporter = ProgressReporter::default();

        let (cb1, log1) = capturing_callback();
        reporter.set_callback(cb1);
        reporter.info("first");
        assert_eq!(log1.lock().unwrap().len(), 1);

        let (cb2, log2) = capturing_callback();
        reporter.set_callback(cb2);
        reporter.info("second");

        assert_eq!(log1.lock().unwrap().len(), 1);
        assert_eq!(log2.lock().unwrap().len(), 1);
        assert!(log2.lock().unwrap()[0].contains("second"));
    }
}
