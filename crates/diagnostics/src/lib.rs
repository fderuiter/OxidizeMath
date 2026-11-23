#![doc = include_str!("../README.md")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Severity {
    #[allow(missing_docs)]
    Info,
    #[allow(missing_docs)]
    Warning,
    #[allow(missing_docs)]
    Error,
    #[allow(missing_docs)]
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

#[allow(missing_docs)]
pub trait Diagnostic: std::error::Error + Send + Sync + 'static {
    #[allow(missing_docs)]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[allow(missing_docs)]
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct DiagnosticEvent {
    #[allow(missing_docs)]
    pub severity: Severity,
    #[allow(missing_docs)]
    pub message: String,
    #[allow(missing_docs)]
    pub metadata: HashMap<String, String>,
    #[allow(missing_docs)]
    pub thread_name: Option<String>,
}

#[allow(missing_docs)]
pub type ListenerFn = Arc<dyn Fn(&DiagnosticEvent) + Send + Sync + 'static>;

#[allow(missing_docs)]
pub struct DiagnosticBus {
    listeners: Arc<Mutex<Vec<ListenerFn>>>,
}

impl DiagnosticBus {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[allow(missing_docs)]
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

    #[allow(missing_docs)]
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

#[allow(missing_docs)]
pub fn global_bus() -> &'static DiagnosticBus {
    static BUS: OnceLock<DiagnosticBus> = OnceLock::new();
    BUS.get_or_init(DiagnosticBus::new)
}

#[allow(missing_docs)]
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
