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

/// Unified error telemetry contract for `#![no_std]` and `std` domain errors.
///
/// Implementors automatically default to [`Severity::Error`], zero-allocation static
/// [`Diagnostic::error_type`] / [`Diagnostic::code`] identifiers, and (when `std` is enabled) a
/// default metadata map containing `"error_type"`, `"description"`, and optional `"code"`.
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
///     fn error_type(&self) -> &'static str {
///         "StepDivergenceError"
///     }
///
///     fn metadata(&self) -> HashMap<String, String> {
///         HashMap::from([("iteration".to_string(), self.iteration.to_string())])
///     }
/// }
///
/// let err = StepDivergenceError { iteration: 42 };
/// assert_eq!(err.severity(), Severity::Warning);
/// assert_eq!(err.error_type(), "StepDivergenceError");
/// assert_eq!(err.metadata().get("iteration").map(String::as_str), Some("42"));
/// ```
pub trait Diagnostic: core::error::Error + Send + Sync + 'static {
    /// Returns the operational [`Severity`] associated with this error (defaults to [`Severity::Error`]).
    fn severity(&self) -> Severity {
        Severity::Error
    }

    /// Returns a zero-allocation static error type identifier suitable for `#![no_std]` kernels.
    fn error_type(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Returns an optional zero-allocation static diagnostic code (defaults to `None`).
    fn code(&self) -> Option<&'static str> {
        None
    }

    /// Returns structured key-value diagnostic attributes describing the failure context.
    #[cfg(feature = "std")]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("description".to_string(), self.to_string());
        if let Some(code) = self.code() {
            map.insert("code".to_string(), code.to_string());
        }
        map
    }
}

#[cfg(feature = "std")]
fn current_thread_label() -> Option<String> {
    let current = std::thread::current();
    Some(
        current
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{:?}", current.id())),
    )
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
    /// Missing `"error_type"`, `"code"`, and `"source"` keys are populated automatically from
    /// [`Diagnostic::error_type`] and [`Diagnostic::code`] so downstream telemetry listeners retain
    /// structured attribution.
    pub fn emit_diagnostic<D: Diagnostic + ?Sized>(&self, diagnostic: &D) {
        let mut metadata = diagnostic.metadata();
        metadata
            .entry("error_type".to_string())
            .or_insert_with(|| diagnostic.error_type().to_string());
        if let Some(code) = diagnostic.code() {
            metadata
                .entry("code".to_string())
                .or_insert_with(|| code.to_string());
        }
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

    /// Converts a [`Diagnostic`] error into a [`DiagnosticEvent`] tagged with an explicit `source`
    /// and dispatches it to all registered listeners.
    pub fn emit_diagnostic_with_source<D: Diagnostic + ?Sized>(
        &self,
        source: &str,
        diagnostic: &D,
    ) {
        assert!(!source.is_empty(), "diagnostic source must not be empty");
        let mut metadata = diagnostic.metadata();
        metadata
            .entry("error_type".to_string())
            .or_insert_with(|| diagnostic.error_type().to_string());
        if let Some(code) = diagnostic.code() {
            metadata
                .entry("code".to_string())
                .or_insert_with(|| code.to_string());
        }
        metadata.insert("source".to_string(), source.to_string());
        self.emit(DiagnosticEvent {
            severity: diagnostic.severity(),
            message: diagnostic.to_string(),
            metadata,
            thread_name: current_thread_label(),
        });
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
            thread_name: current_thread_label(),
        });

        default_hook(panic_info);
    }));
}

#[cfg(all(test, feature = "std"))]
mod tests;
