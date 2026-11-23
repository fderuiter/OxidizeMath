#![doc = include_str!("../README.md")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Standardized operational severity levels for diagnostic events and domain errors.
///
/// # Example
/// ```
/// use diagnostics::Severity;
///
/// assert_eq!(Severity::Info.to_string(), "INFO");
/// assert_eq!(Severity::Warning.to_string(), "WARNING");
/// assert_eq!(Severity::Error.to_string(), "ERROR");
/// assert_eq!(Severity::Fatal.to_string(), "FATAL");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// Informational telemetry indicating normal subsystem progress.
    Info,
    /// Recoverable numerical or operational degradation (for example, clamped time steps).
    Warning,
    /// Domain or solver failure requiring caller handling.
    Error,
    /// Unrecoverable invariant violation or unhandled thread panic.
    Fatal,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Unified error telemetry contract for routing structured domain errors onto a [`DiagnosticBus`].
///
/// Implementors automatically default to [`Severity::Error`] and an empty metadata map unless
/// overridden to attach domain-specific diagnostic context.
///
/// # Example
/// ```
/// use diagnostics::{Diagnostic, Severity};
/// use std::collections::HashMap;
/// use std::fmt;
///
/// #[derive(Debug)]
/// struct StepDivergenceError {
///     iteration: usize,
/// }
///
/// impl fmt::Display for StepDivergenceError {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "solver diverged at iteration {}", self.iteration)
///     }
/// }
///
/// impl std::error::Error for StepDivergenceError {}
///
/// impl Diagnostic for StepDivergenceError {
///     fn severity(&self) -> Severity {
///         Severity::Warning
///     }
///
///     fn metadata(&self) -> HashMap<String, String> {
///         HashMap::from([("iteration".to_string(), self.iteration.to_string())])
///     }
/// }
///
/// let err = StepDivergenceError { iteration: 42 };
/// assert_eq!(err.severity(), Severity::Warning);
/// assert_eq!(err.metadata().get("iteration").map(String::as_str), Some("42"));
/// ```
pub trait Diagnostic: std::error::Error + Send + Sync + 'static {
    /// Returns the operational [`Severity`] associated with this error (defaults to [`Severity::Error`]).
    fn severity(&self) -> Severity {
        Severity::Error
    }

    /// Returns structured key-value diagnostic attributes describing the failure context.
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Structured diagnostic record dispatched through a [`DiagnosticBus`].
#[derive(Debug, Clone)]
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
/// ```
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
///     log_clone.lock().unwrap().push((ev.severity.clone(), ev.message.clone()));
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
}

impl Default for DiagnosticBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a reference to the process-wide singleton [`DiagnosticBus`].
///
/// # Example
/// ```
/// use diagnostics::{global_bus, DiagnosticEvent, Severity};
/// use std::collections::HashMap;
/// use std::sync::{Arc, Mutex};
///
/// let captured = Arc::new(Mutex::new(Vec::new()));
/// let captured_clone = Arc::clone(&captured);
/// global_bus().register_listener(move |ev| {
///     if ev.metadata.get("tag").map(String::as_str) == Some("doctest_global") {
///         captured_clone.lock().unwrap().push(ev.severity.clone());
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
            thread_name: std::thread::current().name().map(|s| s.to_string()),
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
}
