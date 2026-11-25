use super::*;

#[test]
fn test_diagnostic_bus_delivers_events_to_registered_listeners() {
    let bus = DiagnosticBus::default();
    let first_log = Arc::new(Mutex::new(Vec::new()));
    let second_log = Arc::new(Mutex::new(Vec::new()));

    let first_clone = Arc::clone(&first_log);
    bus.register_listener(move |ev| {
        first_clone
            .lock()
            .unwrap()
            .push((ev.severity.clone(), ev.message.clone()));
    });

    let second_clone = Arc::clone(&second_log);
    bus.register_listener(move |ev| {
        second_clone
            .lock()
            .unwrap()
            .push((ev.metadata.get("source").cloned(), ev.thread_name.clone()));
    });

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "solver".to_string());
    bus.emit(DiagnosticEvent {
        severity: Severity::Warning,
        message: "Step clamped".to_string(),
        metadata,
        thread_name: Some("worker-1".to_string()),
    });

    assert_eq!(
        *first_log.lock().unwrap(),
        vec![(Severity::Warning, "Step clamped".to_string())]
    );
    assert_eq!(
        *second_log.lock().unwrap(),
        vec![(Some("solver".to_string()), Some("worker-1".to_string()))]
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_diagnostic_bus_isolates_listener_panics() {
    let bus = DiagnosticBus::new();
    let delivered = Arc::new(Mutex::new(Vec::new()));

    bus.register_listener(|_ev| {
        panic!("intentional listener failure");
    });

    let delivered_clone = Arc::clone(&delivered);
    bus.register_listener(move |ev| {
        delivered_clone.lock().unwrap().push(ev.message.clone());
    });

    bus.emit(DiagnosticEvent {
        severity: Severity::Error,
        message: "Survives preceding panic".to_string(),
        metadata: HashMap::new(),
        thread_name: None,
    });

    assert_eq!(
        *delivered.lock().unwrap(),
        vec!["Survives preceding panic".to_string()]
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_diagnostic_bus_recovers_from_poisoned_listener_lock() {
    let bus = DiagnosticBus::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = bus.listeners.lock().unwrap();
        panic!("poison listeners mutex");
    }));
    assert!(bus.listeners.is_poisoned());

    let received = Arc::new(Mutex::new(Vec::new()));
    let received_clone = Arc::clone(&received);
    bus.register_listener(move |ev| {
        received_clone.lock().unwrap().push(ev.severity.to_string());
    });

    bus.emit(DiagnosticEvent {
        severity: Severity::Fatal,
        message: "Recovered after poison".to_string(),
        metadata: HashMap::new(),
        thread_name: None,
    });

    assert_eq!(*received.lock().unwrap(), vec!["FATAL".to_string()]);
}

#[derive(Debug, thiserror::Error)]
#[error("step divergence at k={step}")]
struct SolverTestError {
    step: usize,
}

impl Diagnostic for SolverTestError {
    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn error_type(&self) -> &'static str {
        "SolverTestError"
    }

    fn code(&self) -> Option<&'static str> {
        Some("SOLVER-001")
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("step".to_string(), self.step.to_string());
        map
    }
}

#[test]
fn test_emit_diagnostic_helpers_populate_source_code_and_thread_label() {
    let bus = DiagnosticBus::new();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    bus.register_listener(move |ev| {
        captured_clone.lock().unwrap().push(ev.clone());
    });

    let err = SolverTestError { step: 7 };
    assert_eq!(err.error_type(), "SolverTestError");
    assert_eq!(err.code(), Some("SOLVER-001"));
    bus.emit_diagnostic(&err);
    bus.emit_diagnostic_with_source("rk4_integrator", &err);

    let events = captured.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].severity, Severity::Warning);
    assert_eq!(events[0].message, "step divergence at k=7");
    assert_eq!(
        events[0].metadata.get("source").map(String::as_str),
        Some("SolverTestError")
    );
    assert_eq!(
        events[0].metadata.get("code").map(String::as_str),
        Some("SOLVER-001")
    );
    assert!(events[0].thread_name.is_some());
    assert_eq!(
        events[1].metadata.get("source").map(String::as_str),
        Some("rk4_integrator")
    );
    assert_eq!(
        events[1].metadata.get("step").map(String::as_str),
        Some("7")
    );
}
