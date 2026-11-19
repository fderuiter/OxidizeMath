//! Thread-tagged telemetry dispatch and test-isolated polling for `MathExplorerApp`.
//!
//! Ensures every GUI telemetry event records its originating thread identity and
//! prevents concurrent test threads' simulation divergence diagnostics from leaking
//! into unrelated `MathExplorerApp::update` accessibility announcements or panel state.

use federated_registry::{global_registry, Severity, TelemetryEvent};
use std::collections::HashMap;

/// Returns a non-empty label identifying the calling thread by name or `ThreadId`.
pub(crate) fn current_thread_label() -> Option<String> {
    let current = std::thread::current();
    Some(
        current
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{:?}", current.id())),
    )
}

/// Emits a structured [`TelemetryEvent`] tagged with the calling thread's identity.
pub(crate) fn emit_telemetry(
    source: &str,
    severity: Severity,
    message: String,
    metadata: HashMap<String, String>,
) {
    global_registry().emit(TelemetryEvent {
        source: source.to_string(),
        severity,
        message,
        metadata,
        thread_name: current_thread_label(),
    });
}

/// Polls pending [`TelemetryEvent`] records destined for `MathExplorerApp::update`.
#[cfg(not(test))]
pub(crate) fn poll_app_telemetry() -> Vec<TelemetryEvent> {
    global_registry().try_recv_all()
}

/// Polls pending [`TelemetryEvent`] records for the active test thread, preserving
/// events emitted by unrelated parallel test threads in the shared registry queue.
#[cfg(test)]
pub(crate) fn poll_app_telemetry() -> Vec<TelemetryEvent> {
    let current = current_thread_label();
    let mut local = Vec::new();
    for ev in global_registry().try_recv_all() {
        if ev.thread_name.is_none() || ev.thread_name == current {
            local.push(ev);
        } else {
            global_registry().emit(ev);
        }
    }
    local
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poll_app_telemetry_isolates_foreign_thread_events() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let _ = global_registry().try_recv_all();

        std::thread::Builder::new()
            .name("foreign_divergence_test_thread".to_string())
            .spawn(|| {
                emit_telemetry(
                    "OdeSolverTool",
                    Severity::Error,
                    "Simulation diverged on foreign thread".to_string(),
                    HashMap::new(),
                );
            })
            .expect("spawn foreign thread")
            .join()
            .expect("join foreign thread");

        emit_telemetry(
            "LocalAppSource",
            Severity::Warning,
            "Same-thread warning".to_string(),
            HashMap::new(),
        );
        global_registry().emit(TelemetryEvent {
            source: "DirectUntaggedTest".to_string(),
            severity: Severity::Info,
            message: "Untagged direct event".to_string(),
            metadata: HashMap::new(),
            thread_name: None,
        });

        let events = poll_app_telemetry();
        assert!(
            !events.iter().any(|e| e.source == "OdeSolverTool"),
            "Foreign test thread telemetry must not leak into poll_app_telemetry"
        );
        assert!(events
            .iter()
            .any(|e| e.source == "LocalAppSource" && e.thread_name == current_thread_label()));
        assert!(events.iter().any(|e| e.source == "DirectUntaggedTest"));
        let requeued = global_registry().try_recv_all();
        assert!(requeued.iter().any(|e| e.source == "OdeSolverTool"));
    }
}
