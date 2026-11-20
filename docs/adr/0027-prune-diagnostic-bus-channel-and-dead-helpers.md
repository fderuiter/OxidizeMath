# ADR 0027: Prune `DiagnosticBus` Unread Channel and Dead Telemetry Helpers

## Status
Accepted (extends [ADR 0016](0016-purge-dead-code-suppressions.md) and [ADR 0019](0019-purge-dead-code-bypasses.md))

## Context
1. **Unread `mpsc` Channel in `DiagnosticBus`**: `diagnostics::DiagnosticBus` allocated an unbounded `std::sync::mpsc::channel()` (`sender` and `receiver`) and cloned every `DiagnosticEvent` into `self.sender` inside `DiagnosticBus::emit` before notifying registered `listeners`. Because `DiagnosticBus::try_recv_all` had zero callers across the workspace—downstream GUI telemetry subscribes via `diagnostics::global_bus().register_listener(...)` inside `federated_registry::global_registry()`—every emitted diagnostic accumulated an unread clone in heap memory for the lifetime of the process.
2. **Uncalled Public Telemetry Helpers**: Four additional helper functions had zero workspace callers, violating the dead-code hygiene policies in [ADR 0016](0016-purge-dead-code-suppressions.md) and [ADR 0019](0019-purge-dead-code-bypasses.md):
   - `DiagnosticBus::emit_error` in `crates/diagnostics/src/lib.rs`
   - `diagnostics::emit` and `diagnostics::emit_error` in `crates/diagnostics/src/lib.rs` (referenced only in stale prose in `crates/diagnostics/README.md`)
   - `federated_registry::init_panic_hook` in `crates/federated_registry/src/lib.rs` (superseded by `diagnostics::init_panic_hook` combined with `federated_registry::global_registry()`)

## Decision
1. **Listener-Only `DiagnosticBus`**: Remove `std::sync::mpsc` imports, `sender`, `receiver`, and `try_recv_all` from `diagnostics::DiagnosticBus` so `DiagnosticBus::emit` dispatches `&DiagnosticEvent` directly to registered `listeners` while preserving poisoned-lock recovery and `std::panic::catch_unwind` isolation per listener:
   ```rust
   use diagnostics::{DiagnosticBus, DiagnosticEvent, Severity};
   use std::collections::HashMap;
   use std::sync::{Arc, Mutex};

   let bus = DiagnosticBus::new();
   let count = Arc::new(Mutex::new(0usize));
   let count_ref = Arc::clone(&count);
   bus.register_listener(move |_| {
       *count_ref.lock().unwrap() += 1;
   });
   bus.emit(DiagnosticEvent {
       severity: Severity::Info,
       message: "direct listener delivery".to_string(),
       metadata: HashMap::new(),
       thread_name: None,
   });
   assert_eq!(*count.lock().unwrap(), 1);
   ```
2. **Remove Uncalled Helpers**: Delete `DiagnosticBus::emit_error`, `diagnostics::emit`, `diagnostics::emit_error`, and `federated_registry::init_panic_hook`, while retaining `Severity`, `Diagnostic`, `DiagnosticEvent`, `ListenerFn`, `global_bus`, `diagnostics::init_panic_hook`, and `federated_registry::global_registry`.
3. **Refresh Baseline and Documentation**: Update `crates/diagnostics/README.md` to document listener registration and `DiagnosticBus::emit`, add unit tests for `DiagnosticBus` and `federated_registry::global_registry`, and regenerate `crates/unified_verification/api_baseline.json`.

## Consequences
- **Positive**: Eliminates redundant `DiagnosticEvent` cloning and unbounded heap growth in `diagnostics::global_bus()` while preserving fatal panic forwarding into `federated_registry::global_registry()` and `math_explorer_gui`.
- **Positive**: Removes five uncalled public signatures from `crates/unified_verification/api_baseline.json` without introducing lint suppressions or `verification_whitelist.toml` entries.
- **Neutral**: Domain `Diagnostic` trait implementations and `FederatedRegistry` polling semantics in `math_explorer_gui` remain unchanged.
