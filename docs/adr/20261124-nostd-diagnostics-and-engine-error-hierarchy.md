# ADR 20261124: `#![no_std]` Diagnostics Trait, Foundation/Engine Error Hierarchy, and Production Diagnostic Telemetry

## Status

Accepted

* **Date:** 2026-11-24

## Context

Following [ADR 20261123](./20261123-hierarchical-umbrella-errors-and-diagnostic-bus-helpers.md), Tier 0 and Tier 2 `#![no_std]` crates (`crates/00_foundation/math_rng` and `crates/02_engines/engine_linalg`) extracted low-discrepancy quasi-Monte Carlo generators (`QmcError`) and dense/sparse matrix solvers (`LinalgError`), while `crates/math_commons` introduced WGSL/GLSL shader lowering (`ShaderEmitError`). Several gaps emerged across the workspace error hierarchy:

1. `crates/diagnostics` unconditionally required `std` (`std::error::Error` and `HashMap<String, String>`), preventing `#![no_std]` crates (`math_rng`, `engine_linalg`, and `math_commons` with `--no-default-features`) from implementing `diagnostics::Diagnostic` or exposing `MathCommonsError` and AST error types in `no_std` builds.
2. `engine_linalg::LinalgError` and `math_rng::QmcError` were absent from `math_commons::error::MathCommonsError`, downstream crate umbrellas, and `math_explorer::error::{Error, Result}`, while `crates/unified_verification/src/api_drift/facade.rs` omitted `crates/00_foundation/math_rng/src/`, `crates/02_engines/engine_linalg/src/`, `crates/engine_diffeq/src/`, and `crates/engine_pde/src/` from `FACADE_REACHABLE_PREFIXES`.
3. Upstream error conversions were incomplete across domain umbrellas (`PureMathError`, `AppliedError`, `PhysicsError`, `BiologyError`, `EpidemiologyDomainError`, `AIError`, and `ClimateError`), and several solvers (`crates/math_commons/src/fit/{lm.rs,poly.rs}`, `crates/pure_math/src/pure_math/algebra/linear_algebra/numerical.rs`, and `crates/pure_math/src/pure_math/statistics/kelly/portfolio.rs`) mapped `LinalgError` into flat leaf variants that discarded `Error::source()`.
4. `DiagnosticBus::emit_diagnostic` and `emit_diagnostic_with_source` had no production callers outside unit tests and left `ModuleDiagnostic::thread_name` unpopulated, risking cross-thread event leakage in GUI telemetry polling.

## Decision

1. **Make `diagnostics::Diagnostic` and `Severity` `#![no_std]`-Compatible**:
   - Add `default = ["std"]`, `alloc = []`, and `std = ["alloc"]` features to `crates/diagnostics`.
   - Bound `Diagnostic` on `core::error::Error + Send + Sync + 'static` with zero-allocation `severity(&self) -> Severity`, `error_type(&self) -> &'static str`, and `code(&self) -> Option<&'static str>` methods available in `#![no_std]`, while gating `HashMap`-backed `metadata(&self)`, `ModuleDiagnostic`, `DiagnosticBus`, and `global_bus()` behind `#[cfg(feature = "std")]`.
   - Implement `Diagnostic` for `math_rng::QmcError` (`RNG-QMC-001`..`RNG-QMC-004`) and `engine_linalg::LinalgError` (`ENGINE-LINALG-001`..`ENGINE-LINALG-006`), and ungate `math_commons::error` (`MathCommonsError`, `FitError`, `MathError`, `EvalError`, `GlslError`, `LexError`, `ParseError`, `ShaderEmitError`) for `#![no_std]` + `alloc` builds.

2. **Complete Foundation, Engine, Crate Umbrella, and Facade Error Conversions**:
   - Re-export `LinalgError` and `QmcError` in `math_commons::error`, `math_explorer::error`, and `math_explorer::prelude`, with direct zero-allocation `From` conversions into `MathCommonsError`, `PureMathError`, `AppliedError`, `PhysicsError`, `BiologyError`, `EpidemiologyDomainError`, `AIError`, `ClimateError`, and `math_explorer::error::Error`.
   - Add missing cross-layer `From` conversions on `PureMathError` (`FitError`, `ShaderEmitError`), `AppliedError` (`FitError`, `MonteCarloError`, `ShaderEmitError`), `PhysicsError` (`FitError`, `TensorError`, `ShaderEmitError`), `BiologyError` & `EpidemiologyDomainError` (`FitError`, `LinearAlgebraError`, AST errors), and `AIError` & `ClimateError` (`OptimizationError`, `MathCommonsError`, `FitError`, AST errors), marking `AIError` and `ClimateError` `#[non_exhaustive]`.

3. **Preserve Causal `Error::source()` Chains Across Linear Algebra, Curve Fitting, Kelly, and QMC**:
   - Add `FitError::Linalg(#[from] LinalgError)`, `LinearAlgebraError::Linalg(#[from] LinalgError)`, `KellyError::Linalg(LinalgError)` (with `KellyError::NotPositiveDefinite` also returning `Some(&LinalgError::NotPositiveDefinite)` from `Error::source()`), and `MonteCarloError::Qmc(#[from] QmcError)` so upstream `#![no_std]` failure causes remain inspectable via `Error::source()` without heap allocation ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

4. **Wire Thread-Tagged Diagnostic Emission and Extend `api_drift` Facade Auditing**:
   - Tag `DiagnosticBus::emit_diagnostic` and `emit_diagnostic_with_source` events with the calling thread's label (`name` or `ThreadId`), expose `math_explorer::error::Error::{emit_to_global_bus, emit_with_source}` and `math_explorer_gui::telemetry::emit_diagnostic`, and emit structured `ClimateError` diagnostics from `CeraTab` on model initialization failures.
   - Add `crates/00_foundation/math_rng/src/`, `crates/02_engines/engine_linalg/src/`, `crates/engine_diffeq/src/`, and `crates/engine_pde/src/` to `FACADE_REACHABLE_PREFIXES` in `crates/unified_verification/src/api_drift/facade.rs` while ignoring sealed `private::*` supertrait bounds.

## Consequences

* **Positive:** Every `#![no_std]` foundation and engine error implements `diagnostics::Diagnostic`, converts in one zero-allocation `?` hop into crate umbrellas and `math_explorer::error::Result`, and preserves `Error::source()` chains from Tier 0/Tier 2 kernels up to the SDK facade and GUI telemetry bus.
* **Positive:** `unified_verification` automatically rejects any future un-reexported `*Error` type introduced in `math_rng`, `engine_linalg`, `engine_diffeq`, or `engine_pde`.
* **Trade-offs:** Public error additions across Tier 0 and Tier 2 engine crates now participate in facade completeness checks and `api_baseline.json` tracking.
