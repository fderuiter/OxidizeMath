# ADR 20261124: `#![no_std]` Visitor-Based `math_diagnostics`, Causal `LinalgError` Retention, and Umbrella `api_drift` Guards

## Status

Accepted

* **Date:** 2026-11-24

## Context

OxidizeMath's Tier 2 numerical engines (`engine_linalg`, `engine_diffeq`, and `engine_pde`) and `math_commons` (`default-features = false`) target `#![no_std]` execution on `wasm32-unknown-unknown` (see [ADR 20261123](./20261123-no-std-engine-linalg-and-solver-unification.md) and [ADR 20261123](./20261123-nostd-diffeq-and-pde-engines.md)). However, five structural gaps prevented uniform zero-allocation diagnostics and end-to-end causal error propagation across workspace tiers:

1. **`std`-Only Heap-Allocating `Diagnostic` Trait**: `crates/diagnostics` required `std` and defined `Diagnostic::metadata(&self) -> HashMap<String, String>`, preventing `#![no_std]` crates (`engine_linalg` and `math_commons` without `std`) from implementing `Diagnostic` and allocating a `HashMap` plus `String` entries on every diagnostic inspection.
2. **Dropped `LinalgError` Causal Chains and `std`-Gated PDE Helpers**: `FitError` and `LinearAlgebraError` discarded `engine_linalg::LinalgError` via `.map_err(...)` instead of wrapping it with `#[from]`, leaving `Error::source()` as `None`, while `engine_pde` gated `StencilOperators::try_new*` and `FusedStencilStepper::try_new*` behind `#[cfg(feature = "std")]` even though `MathError` is `core`-compatible.
3. **19 Cross-Crate Leaf-Error Call Sites Missing Crate Umbrella Conversions**: Methods in `domain_ai`, `domain_climate`, `domain_applied`, and `domain_physics` returning `OptimizationError`, `FitError`, `MonteCarloError`, or `TensorError` could not be composed with `?` into their owning crate's error type (`AIError`, `ClimateError`, `AppliedError` / `BatteryError` / `GameTheoryError`, and `PhysicsError` / `SandpileError` / `HighEnergyError`).
4. **Bypassed `DiagnosticBus` in `math_explorer_gui`**: `math_explorer_gui::telemetry` wrote `TelemetryEvent` structs directly to `federated_registry::global_registry()` rather than dispatching through `DiagnosticBus::emit_diagnostic_with_source`.
5. **Unchecked Tier 2 Engine & Umbrella `From` Completeness in `api_drift`**: `crates/unified_verification/src/api_drift` audited `math_explorer/src/error.rs` re-exports for domain crates (`ADR 20261123`), but did not inspect Tier 2 engine error definitions or enforce crate umbrella `From` conversions.

## Decision

1. **Extract Tier 0 `#![no_std]` `crates/00_foundation/math_diagnostics`**:
   - Define `Severity`, `DiagnosticVisitor`, and `Diagnostic: core::error::Error + Send + Sync + 'static` using zero-allocation `visit_fields(&self, visitor: &mut dyn DiagnosticVisitor)` without `std` or `alloc` when `default-features = false`.
   - Provide a backward-compatible default `Diagnostic::metadata(&self) -> HashMap<String, String>` under the optional `std` feature via `collect_diagnostic_metadata`.
   - Adapt `crates/diagnostics` as the `std` event-bus bridge re-exporting `Diagnostic`, `DiagnosticVisitor`, `Severity`, and `collect_diagnostic_metadata`, materializing `HashMap<String, String>` inside `DiagnosticBus::emit_diagnostic` and `emit_diagnostic_with_source` while tagging unnamed threads with their `ThreadId`.

2. **Implement Zero-Allocation `Diagnostic` Across Engines, `math_commons`, `pure_math`, `domain_*`, and `math_explorer`**:
   - Implement `#![no_std]` `Diagnostic` on `engine_linalg::LinalgError` and unconditionally in `math_commons` (`MathError`, `FitError`, `MathCommonsError`, plus `std`-gated AST errors).
   - Ungate `StencilOperators::{try_new, try_new_2d, try_new_3d}` and `FusedStencilStepper::{try_new, try_new_2d, try_new_3d}` in `crates/engine_pde` so `#![no_std]` callers have full checked parity.
   - Migrate all library `Diagnostic` implementations in `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, `domain_physics`, and `math_explorer` to `visit_fields`.

3. **Preserve `LinalgError` Causal `source()` Chains and Cross-Crate `?` Conversions**:
   - Add `#[from] LinalgError` to `FitError`, `LinearAlgebraError`, `MathCommonsError`, `PureMathError`, all 6 domain crate umbrellas (`AIError`, `AppliedError`, `BiologyError`, `ClimateError`, `EpidemiologyDomainError`, `PhysicsError`), and `math_explorer::error::Error`, replacing lossy `.map_err(...)` wrappers in `math_commons::fit` and `pure_math::algebra::linear_algebra::numerical` with `?`.
   - Wire `From<OptimizationError>` on `AIError` and `ClimateError`, `From<FitError>` on `BatteryError` / `SandpileError` / `AppliedError` / `PhysicsError`, `From<MonteCarloError>` on `GameTheoryError` / `AppliedError`, and `From<TensorError>` on `HighEnergyError` / `PhysicsError` so all 19 cross-crate methods convert via `?`.

4. **Bridge `math_explorer_gui` Telemetry Through `DiagnosticBus::emit_diagnostic_with_source`**:
   - Route `math_explorer_gui::telemetry::emit_diagnostic` and `emit_telemetry` through `diagnostics::global_bus().emit_diagnostic_with_source(...)` into `federated_registry::global_registry()`, preserving per-thread isolation in `poll_app_telemetry()`.

5. **Enforce Tier 2 Engine Re-Exports and Crate Umbrella `From` Conversions in `api_drift`**:
   - Extend `crates/unified_verification/src/api_drift/facade_errors.rs` to verify that every public `*Error` in Tier 2 engines (`engine_linalg`, `engine_diffeq`, `engine_pde`) and domain crates is re-exported by `math_explorer/src/error.rs` AND has a `From` conversion into its crate umbrella error (including cross-crate `Result<_, E>` return types).

## Consequences

* **Positive:** `math_diagnostics` (without `alloc`), `engine_linalg`, `engine_diffeq`, `engine_pde`, and `math_commons` (`--no-default-features`) compile cleanly on `wasm32-unknown-unknown` without `std`, and visiting `Diagnostic` fields performs zero heap allocations under `VerifiedEngine::run_verified`.
* **Positive:** Every `LinalgError` and cross-crate solver error retains its `Error::source()` chain through leaf, crate-umbrella, and `math_explorer::error::Error` conversions while flowing through `DiagnosticBus` to `federated_registry`.
* **Trade-offs:** Adding a new Tier 2 engine error or returning an external error from a domain crate's public signature now requires adding the corresponding `math_explorer::error` re-export and crate umbrella `From` conversion to pass `unified_verification check-api-drift`.
