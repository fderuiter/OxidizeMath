# Federated Registry

Decentralized telemetry bus, event streaming, and model discovery registry for the OxidizeMath ecosystem.

## 1. Overview & Architecture

The `federated_registry` crate provides a lightweight, thread-safe event bus and telemetry routing framework. It coordinates diagnostics, warning signals, and status notifications across decoupled mathematical kernels and visualization layers.

By utilizing non-blocking mpsc channels and isolated source catalogs, the registry ensures that high-throughput simulation workers can emit telemetry events without contending for global locks or introducing direct dependencies between worker threads and graphical frontends.

### Module Structure

- **`FederatedRegistry`**: The primary event registry managing telemetry dispatch channels and registered subsystem sources.
  - Supports dynamic source registration (`register_source`, `known_sources`).
  - Implements thread-safe non-blocking event dispatch (`emit`) and batched reception (`try_recv_all`).
- **`TelemetryEvent`**: Structured telemetry record capturing:
  - `source`: The originating subsystem or simulation identifier.
  - `severity`: Standardized diagnostic severity levels (`Info`, `Warning`, `Error`, `Fatal`).
  - `message`: Human-readable operational description.
  - `metadata`: Key-value attribute map for contextual metrics.
  - `thread_name`: Optional thread identity for concurrent diagnostics.
- **`global_registry`**: Global thread-safe singleton integrating `diagnostics::global_bus` with the federated telemetry channel.

---

## 2. Developer Instructions

### Registering Sources and Emitting Telemetry

Subsystems register their source identity and emit telemetry records through local or global registry instances:

```rust
use federated_registry::{FederatedRegistry, Severity, TelemetryEvent};
use std::collections::HashMap;

// Initialize a local telemetry registry instance
let registry = FederatedRegistry::new();

// Register the simulation source
registry.register_source("n_body_simulation");
assert!(registry.known_sources().contains(&"n_body_simulation".to_string()));

// Construct a structured telemetry event
let mut metadata = HashMap::new();
metadata.insert("particle_count".to_string(), "5000".to_string());
metadata.insert("timestep".to_string(), "0.001".to_string());

let event = TelemetryEvent {
    source: "n_body_simulation".to_string(),
    severity: Severity::Info,
    message: "Initial conditions converged".to_string(),
    metadata,
    thread_name: Some("worker_0".to_string()),
};

// Dispatch event to the registry
registry.emit(event);

// Retrieve and inspect dispatched events
let received_events = registry.try_recv_all();
assert_eq!(received_events.len(), 1);
assert_eq!(received_events[0].source, "n_body_simulation");
assert_eq!(received_events[0].severity, Severity::Info);
assert_eq!(received_events[0].message, "Initial conditions converged");
```

---

## 3. Ecosystem Integration

The `federated_registry` crate acts as an aggregation bridge between low-level simulation workers and the `math_explorer_gui` notification panels, ensuring unified observability without runtime coupling.
