# Architecture Decision Record 0007: Reflective UI Fallback Constraints and Telemetry

## Context
In `math_explorer_gui/src/reflective_ui.rs`, missing parameter metadata or constraints in theory definitions resulted in explicit panics (`.unwrap()` or `panic!`). This caused UI thread crashes when rendering theory parameters for unconstrained models or models with missing theory parameter definitions.

## Decision
We refactored reflective parameter rendering in `math_explorer_gui` to gracefully handle missing parameter constraints:
1. Replaced `.unwrap()` and explicit panics with safe fallback parameter bounds (`min: 0.0, max: 100.0, step: 0.1`).
2. Logged diagnostic warnings (`Severity::Warning`) to `federated_registry::global_registry()` whenever fallback constraints are applied.
3. Added unit tests in `math_explorer_gui/src/reflective_ui.rs` to verify non-panicking UI parameter rendering and telemetry warning emission when metadata is missing.

## Consequences
- **Positive:** UI rendering thread does not crash on missing theory parameter metadata or constraint definitions.
- **Positive:** Diagnostic warning events are captured in the global federated registry for debugging missing model metadata.
- **Positive:** Passes architectural verification standards enforced by `unified_verification`.
