#![doc = include_str!("../README.md")]

pub use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity, collect_diagnostic_metadata};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn current_thread_label() -> Option<String> {
    let current = std::thread::current();
    Some(
        current
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{:?}", current.id())),
    )
}

/// Typed stack-resident error returned by fallible [`DiagnosticBus`] emission helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticError {
    /// Explicit diagnostic `source` tag was empty.
    EmptySource,
}

impl std::fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptySource => f.write_str("Diagnostic source tag must not be empty"),
        }
    }
}

impl std::error::Error for DiagnosticError {}

impl Diagnostic for DiagnosticError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "DiagnosticError");
        visitor.record_str("code", "DIAG-001");
        visitor.record_display("description", self);
    }
}

/// Structured diagnostic record dispatched through a [`DiagnosticBus`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticEvent {
    /// Operational severity classification of the event.
    pub severity: Severity,
    /// Human-readable summary of the diagnostic condition.
    pub message: String,
    /// Arbitrary key-value pairs supplying subsystem, location, or parameter context.
    pub metadata: HashMap<String, String>,
    /// Optional name of the originating OS or worker thread.
    pub thread_name: Option<String>,
}

/// Thread-safe callback signature invoked synchronously when a [`DiagnosticBus`] emits an event.
pub type ListenerFn = Arc<dyn Fn(&DiagnosticEvent) + Send + Sync + 'static>;

/// Thread-safe synchronous pub-sub bus for broadcasting [`DiagnosticEvent`] records.
///
/// # Invariants
/// - **Poison Recovery**: Listener registration and event emission recover automatically if a
///   previous holder poisoned the internal listener mutex.
/// - **Panic Isolation**: Each listener invocation in [`DiagnosticBus::emit`] is wrapped in
///   [`std::panic::catch_unwind`] so a faulty subscriber cannot abort delivery to later listeners.
///
/// # Example
/// ```rust
/// use diagnostics::{DiagnosticBus, DiagnosticEvent, Severity};
/// use std::collections::HashMap;
/// use std::sync::{Arc, Mutex};
///
/// let bus = DiagnosticBus::new();
/// let log = Arc::new(Mutex::new(Vec::new()));
///
/// // Even if an earlier listener panics, subsequent listeners still receive the event.
/// bus.register_listener(|_ev| panic!("faulty subscriber"));
///
/// let log_clone = Arc::clone(&log);
/// bus.register_listener(move |ev| {
///     log_clone.lock().unwrap().push((ev.severity, ev.message.clone()));
/// });
///
/// bus.emit(DiagnosticEvent {
///     severity: Severity::Error,
///     message: "Matrix singular".to_string(),
///     metadata: HashMap::new(),
///     thread_name: Some("solver-0".to_string()),
/// });
///
/// assert_eq!(
///     *log.lock().unwrap(),
///     vec![(Severity::Error, "Matrix singular".to_string())]
/// );
/// ```
pub struct DiagnosticBus {
    listeners: Arc<Mutex<Vec<ListenerFn>>>,
}

impl std::fmt::Debug for DiagnosticBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let listener_count = match self.listeners.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        };
        f.debug_struct("DiagnosticBus")
            .field("listener_count", &listener_count)
            .finish()
    }
}

impl DiagnosticBus {
    /// Creates an empty [`DiagnosticBus`] with no registered listeners.
    pub fn new() -> Self {
        Self {
            listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Appends a thread-safe subscriber callback that will be notified on every [`DiagnosticBus::emit`] call.
    pub fn register_listener<F>(&self, listener: F)
    where
        F: Fn(&DiagnosticEvent) + Send + Sync + 'static,
    {
        let mut listeners = match self.listeners.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        listeners.push(Arc::new(listener));
    }

    /// Broadcasts `event` synchronously to all registered listeners while isolating individual listener panics.
    pub fn emit(&self, event: DiagnosticEvent) {
        let listeners = match self.listeners.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for listener in listeners.iter() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                listener(&event);
            }));
            if result.is_err() {
                eprintln!(
                    "Diagnostic listener panicked while processing event: {}",
                    event.message
                );
            }
        }
    }

    /// Materializes a [`Diagnostic`] error into a [`DiagnosticEvent`] and dispatches it to all listeners.
    ///
    /// If visited diagnostic fields omit `"source"` and provide `"error_type"`, `"source"` is
    /// populated automatically so downstream telemetry listeners retain structured attribution.
    pub fn emit_diagnostic<D: Diagnostic + ?Sized>(&self, diagnostic: &D) {
        let mut metadata = diagnostic.metadata();
        if !metadata.contains_key("source")
            && let Some(err_type) = metadata.get("error_type").cloned()
        {
            metadata.insert("source".to_string(), err_type);
        }
        self.emit(DiagnosticEvent {
            severity: diagnostic.severity(),
            message: diagnostic.to_string(),
            metadata,
            thread_name: current_thread_label(),
        });
    }

    /// Fallible companion to [`Self::emit_diagnostic_with_source`] returning
    /// [`DiagnosticError::EmptySource`] when `source` is empty instead of panicking.
    pub fn try_emit_diagnostic_with_source<D: Diagnostic + ?Sized>(
        &self,
        source: &str,
        diagnostic: &D,
    ) -> Result<(), DiagnosticError> {
        if source.is_empty() {
            return Err(DiagnosticError::EmptySource);
        }
        let mut metadata = diagnostic.metadata();
        metadata.insert("source".to_string(), source.to_string());
        self.emit(DiagnosticEvent {
            severity: diagnostic.severity(),
            message: diagnostic.to_string(),
            metadata,
            thread_name: current_thread_label(),
        });
        Ok(())
    }

    /// Materializes a [`Diagnostic`] error into a [`DiagnosticEvent`] tagged with an explicit `source`
    /// and dispatches it to all registered listeners.
    ///
    /// # Panics
    ///
    /// Panics if `source` is empty. Prefer [`Self::try_emit_diagnostic_with_source`] when `source`
    /// originates from dynamic input.
    pub fn emit_diagnostic_with_source<D: Diagnostic + ?Sized>(
        &self,
        source: &str,
        diagnostic: &D,
    ) {
        self.try_emit_diagnostic_with_source(source, diagnostic)
            .expect("diagnostic source must not be empty");
    }
}

impl Default for DiagnosticBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a reference to the process-wide singleton [`DiagnosticBus`].
///
/// # Example
/// ```rust
/// use diagnostics::{global_bus, DiagnosticEvent, Severity};
/// use std::collections::HashMap;
/// use std::sync::{Arc, Mutex};
///
/// let captured = Arc::new(Mutex::new(Vec::new()));
/// let captured_clone = Arc::clone(&captured);
/// global_bus().register_listener(move |ev| {
///     if ev.metadata.get("tag").map(String::as_str) == Some("doctest_global") {
///         captured_clone.lock().unwrap().push(ev.severity);
///     }
/// });
///
/// global_bus().emit(DiagnosticEvent {
///     severity: Severity::Info,
///     message: "Global heartbeat".to_string(),
///     metadata: HashMap::from([("tag".to_string(), "doctest_global".to_string())]),
///     thread_name: None,
/// });
///
/// assert_eq!(*captured.lock().unwrap(), vec![Severity::Info]);
/// ```
pub fn global_bus() -> &'static DiagnosticBus {
    static BUS: OnceLock<DiagnosticBus> = OnceLock::new();
    BUS.get_or_init(DiagnosticBus::new)
}

/// Installs a process-wide panic hook that emits a [`Severity::Fatal`] [`DiagnosticEvent`] on
/// [`global_bus`] (including `location` and thread identity) before delegating to the prior panic hook.
pub fn init_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let mut metadata = HashMap::new();
        metadata.insert("location".to_string(), location);

        global_bus().emit(DiagnosticEvent {
            severity: Severity::Fatal,
            message: format!("Thread panicked: {}", message),
            metadata,
            thread_name: current_thread_label(),
        });

        default_hook(panic_info);
    }));
}

#[cfg(test)]
mod tests {
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
                .push((ev.severity, ev.message.clone()));
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

        fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
            visitor.record_str("error_type", "SolverTestError");
            visitor.record_usize("step", self.step);
        }
    }

    #[test]
    fn test_emit_diagnostic_helpers_populate_source_and_metadata() {
        let bus = DiagnosticBus::new();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        bus.register_listener(move |ev| {
            captured_clone.lock().unwrap().push(ev.clone());
        });

        let err = SolverTestError { step: 7 };
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
            events[1].metadata.get("source").map(String::as_str),
            Some("rk4_integrator")
        );
        assert_eq!(
            events[1].metadata.get("step").map(String::as_str),
            Some("7")
        );
        assert!(events[0].thread_name.is_some());
    }

    #[test]
    fn test_try_emit_diagnostic_with_source_rejects_empty_source() {
        let bus = DiagnosticBus::new();
        let err = SolverTestError { step: 3 };
        assert_eq!(
            bus.try_emit_diagnostic_with_source("", &err),
            Err(DiagnosticError::EmptySource)
        );
        let meta = DiagnosticError::EmptySource.metadata();
        assert_eq!(
            meta.get("error_type").map(String::as_str),
            Some("DiagnosticError")
        );
        assert_eq!(meta.get("code").map(String::as_str), Some("DIAG-001"));
    }
}
