# Diagnostics

High-integrity diagnostic trait, panic-isolated event bus, and thread panic telemetry hooks for the OxidizeMath ecosystem.

## 1. Overview & Architecture

The `diagnostics` crate provides a unified `Diagnostic` error contract and centralized `DiagnosticBus` reporting channel to eliminate silent worker-thread failures and standardize structured error telemetry across all workspace crates.

### Module Structure

- **`Severity`**: Standardized operational severity levels (`Info`, `Warning`, `Error`, `Fatal`) with uppercase `Display` formatting.
- **`Diagnostic`**: Trait extending `std::error::Error + Send + Sync + 'static` with structured `severity()` and key-value `metadata()` inspection.
- **`DiagnosticEvent`**: Thread-aware diagnostic payload carrying `severity`, human-readable `message`, `metadata` map, and optional `thread_name`.
- **`DiagnosticBus` & `ListenerFn`**: Thread-safe synchronous pub-sub bus that recovers from poisoned listener mutexes and isolates panicking subscriber callbacks via `std::panic::catch_unwind`.
- **`global_bus` & `init_panic_hook`**: Process-wide `OnceLock<DiagnosticBus>` singleton and panic hook forwarding unhandled thread panics (with `file:line` location metadata) as `Severity::Fatal` events before invoking the default panic hook.

---

## 2. Developer Instructions

### Implementing `Diagnostic` and Subscribing to `DiagnosticBus`

When defining a domain error type, implement `Diagnostic` so structured context can be routed through local or global diagnostic buses:

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

---

## 3. Ecosystem Integration

`diagnostics` sits at the base of the OxidizeMath dependency graph alongside `federated_registry`. `federated_registry::global_registry()` automatically attaches a listener to `diagnostics::global_bus()`, and `math_explorer_gui` calls `diagnostics::init_panic_hook()` at startup so background simulation panics surface cleanly in UI telemetry panels.
