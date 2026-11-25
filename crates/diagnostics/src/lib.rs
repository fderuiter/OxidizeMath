#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl core::fmt::Display for Severity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Zero-allocation `#![no_std]` telemetry contract for stack-allocated core and engine errors.
pub trait CoreDiagnostic: core::error::Error + Send + Sync + 'static {
    /// Returns the operational [`Severity`] associated with this error (defaults to [`Severity::Error`]).
    fn severity(&self) -> Severity {
        Severity::Error
    }

    /// Stable static error type identifier (for example, `"MathError"` or `"PdeError"`).
    fn error_type(&self) -> &'static str;

    /// Stable static diagnostic code (for example, `"MATH-001"` or `"PDE-001"`).
    fn code(&self) -> &'static str;
}

/// Typed stack error returned by fallible diagnostic bus operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticError {
    /// Explicit diagnostic source attribution string was empty.
    EmptySource,
}

impl core::fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptySource => f.write_str("diagnostic source must not be empty"),
        }
    }
}

impl core::error::Error for DiagnosticError {}

impl CoreDiagnostic for DiagnosticError {
    fn error_type(&self) -> &'static str {
        "DiagnosticError"
    }

    fn code(&self) -> &'static str {
        match self {
            Self::EmptySource => "DIAG-001",
        }
    }
}

#[cfg(feature = "std")]
impl Diagnostic for DiagnosticError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("code".to_string(), self.code().to_string());
        map.insert("description".to_string(), self.to_string());
        map
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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
pub struct DiagnosticBus {
    listeners: Arc<Mutex<Vec<ListenerFn>>>,
}

#[cfg(feature = "std")]
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

    /// Converts a [`Diagnostic`] error into a [`DiagnosticEvent`] and dispatches it to all listeners.
    ///
    /// If `diagnostic.metadata()` omits `"source"` and provides `"error_type"`, `"source"` is
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
            thread_name: std::thread::current().name().map(str::to_string),
        });
    }

    /// Fallibly converts a [`Diagnostic`] error into a [`DiagnosticEvent`] tagged with `source`
    /// and dispatches it to all registered listeners.
    ///
    /// # Errors
    /// Returns [`DiagnosticError::EmptySource`] if `source` is empty.
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
            thread_name: std::thread::current().name().map(str::to_string),
        });
        Ok(())
    }

    /// Converts a [`Diagnostic`] error into a [`DiagnosticEvent`] tagged with an explicit `source`
    /// and dispatches it to all registered listeners.
    ///
    /// # Panics
    /// Panics if `source` is empty; prefer [`Self::try_emit_diagnostic_with_source`] for fallible callers.
    pub fn emit_diagnostic_with_source<D: Diagnostic + ?Sized>(
        &self,
        source: &str,
        diagnostic: &D,
    ) {
        self.try_emit_diagnostic_with_source(source, diagnostic)
            .expect("diagnostic source must not be empty");
    }
}

#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
pub fn global_bus() -> &'static DiagnosticBus {
    static BUS: OnceLock<DiagnosticBus> = OnceLock::new();
    BUS.get_or_init(DiagnosticBus::new)
}

/// Installs a process-wide panic hook that emits a [`Severity::Fatal`] [`DiagnosticEvent`] on
/// [`global_bus`] (including `location` and thread identity) before delegating to the prior panic hook.
#[cfg(feature = "std")]
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
mod tests;
