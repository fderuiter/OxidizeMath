High-Integrity Diagnostic Suite

This module provides a unified diagnostic trait and centralized reporting bus
to eliminate silent thread failures and standardize error reporting across all modules.

# Developer Guide: Implementing Diagnostics

When creating a new error type in a library module, you must implement the
`Diagnostic` trait to ensure it integrates with the centralized reporting bus.

## Example
```rust
use diagnostics::{Diagnostic, DiagnosticBus, DiagnosticEvent, Severity};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyModuleError {
    #[error("Numerical instability detected at state {state}")]
    Instability { state: String },
}

impl Diagnostic for MyModuleError {
    fn severity(&self) -> Severity {
        match self {
            Self::Instability { .. } => Severity::Warning,
        }
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        match self {
            Self::Instability { state } => {
                meta.insert("simulation_state".to_string(), state.clone());
            }
        }
        meta
    }
}

let err = MyModuleError::Instability {
    state: "t=1.0".to_string(),
};
let bus = DiagnosticBus::new();
let observed = Arc::new(Mutex::new(Vec::new()));
let observed_clone = Arc::clone(&observed);
bus.register_listener(move |event| {
    observed_clone.lock().unwrap().push(event.message.clone());
});

bus.emit(DiagnosticEvent {
    severity: err.severity(),
    message: err.to_string(),
    metadata: err.metadata(),
    thread_name: None,
});
assert_eq!(observed.lock().unwrap().len(), 1);
```

Subscribe downstream consumers with `DiagnosticBus::register_listener` and dispatch structured `DiagnosticEvent` records via `DiagnosticBus::emit` (or `diagnostics::global_bus().emit(...)` for process-wide panic and telemetry forwarding).
