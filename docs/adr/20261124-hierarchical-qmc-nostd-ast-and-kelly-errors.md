# ADR 20261124: Tier 0 `QmcError` Diagnostics, `#![no_std]` AST Error Parity, and `KellyError` Causal Chains

## Status

Accepted

## Context

Following [ADR 20261124: `#![no_std]` Visitor-Based `math_diagnostics`, Causal `LinalgError` Retention, and Umbrella `api_drift` Guards](20261124-no-std-math-diagnostics-and-causal-error-wiring.md) and [ADR 20261124: Optional `#![no_std]` Serde Across Substrate, Plot Outputs, and Facade](20261124-optional-no-std-serde-substrate-and-facade.md), four gaps remained in the cross-crate error hierarchy:

1. **Tier 0 `math_rng::QmcError` diagnostics and umbrella conversions**: `QmcError` lacked `Hash`, `math_diagnostics::Diagnostic::visit_fields`, `MonteCarloError::Qmc`, `MathCommonsError::Qmc`, and `From<QmcError>` conversions across crate umbrellas and `math_explorer::error`.
2. **`math_commons` `--no-default-features` AST and `MathCommonsError` parity**: `math_commons::ast` and the five AST variants of `MathCommonsError` (`Lex`, `Parse`, `Eval`, `Glsl`, `ShaderEmit`) were gated behind `#[cfg(feature = "std")]`, causing `MathCommonsError` to expose different variants between `no_std + alloc` and `std` builds.
3. **`KellyError` causal `Error::source()` retention**: `pure_math::statistics::kelly::portfolio` mapped all `engine_linalg::LinalgError` failures to a unit `KellyError::NotPositiveDefinite` variant whose `Error::source()` returned `None`.
4. **Umbrella extensibility and GUI production telemetry**: `AIError` and `ClimateError` lacked `#[non_exhaustive]` and `MathCommons(#[from] MathCommonsError)` parity, while `Cera` initialization and `CustomFitBridge` formula compilation swallowed structured `ClimateError`/`ParseError`/`EvalError` diagnostics before returning plain `String` UI messages.

## Decision

1. **Tier 0 `QmcError` `Diagnostic` & Full Umbrella Propagation**:
   - Wire `math_rng` directly to Tier 0 `math_diagnostics` (`default-features = false`), derive `Hash` on `QmcError`, and implement zero-allocation `Diagnostic::visit_fields` with stable `RNG-QMC-001..004` codes.
   - Expose `QmcError` across `math_commons::error` (`MathCommonsError::Qmc`), `MonteCarloError::Qmc` (`Copy` preserved), all seven domain/pure-math crate umbrellas (`PureMathError`, `AIError`, `AppliedError`, `BiologyError`, `ClimateError`, `EpidemiologyDomainError`, `PhysicsError`), and `math_explorer::error::Error::Qmc`.
2. **`#![no_std]` (`alloc` + `libm`) AST & Unconditional `MathCommonsError` Variants**:
   - Port `math_commons::ast::{compiled, compiled_serde, eval, expr, glsl, lexer, parser, parser_diagnostics, shader}` to `core` + `alloc` + `libm::atan2`, gating only `HashMap`-backed evaluation helpers and `codegen`/`latex` behind `#[cfg(feature = "std")]`.
   - Expose all `MathCommonsError` variants unconditionally across `no_std` and `std` builds (including `--no-default-features --features serde`), and record `MATH-CORE-001..006` codes on `MathError`.
3. **Causal `Error::source()` on `KellyError` & Modular Error Splits**:
   - Extract `crates/pure_math/src/error/kelly.rs`, `crates/math_commons/src/ast/shader_diagnostics.rs`, `crates/domain_ai/src/error/conversions.rs`, and `crates/domain_climate/src/error/conversions.rs` to isolate conversion/diagnostic boilerplate from enum definitions.
   - Add `KellyError::Linalg(engine_linalg::LinalgError)` and return `Some(&LinalgError::NotPositiveDefinite)` from `KellyError::NotPositiveDefinite.source()`.
4. **Facade Bus Helpers, GUI Telemetry, & `api_drift` Tier 0 Guards**:
   - Add `Error::{emit_to_global_bus, emit_with_source}` on `math_explorer::error::Error` and emit structured diagnostics in `math_explorer_gui`'s `CeraEmulator` and `CustomFitBridge`.
   - Extend `unified_verification::api_drift` to enforce Tier 0 `crates/00_foundation/` error re-exports and umbrella `From` conversions, follow submodule `conversions.rs` files, and ignore `private::*` sealed supertrait bounds.

## Consequences

- `math_rng` and `math_commons` compile and serialize cleanly under `--no-default-features` and `--no-default-features --features serde` with identical `MathCommonsError` variants and zero-allocation `visit_fields`.
- Every Tier 0 (`QmcError`) and Tier 2 (`LinalgError`) error converts via `?` into every crate umbrella and `math_explorer::error::Error` with `Error::source()` intact, enforced statically by `check-api-drift`.
