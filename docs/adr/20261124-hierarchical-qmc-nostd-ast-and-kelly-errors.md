# ADR 20261124: Tier 0 RNG/QMC Diagnostics, Alloc-Gated `#![no_std]` AST, and Causal Kelly Error Wiring

## Status
Accepted (extends [ADR 20261124](20261124-math-rng-static-qmc-views.md), [ADR 20261124](20261124-no-std-math-diagnostics-and-causal-error-wiring.md), [ADR 20261124](20261124-optional-no-std-serde-substrate-and-facade.md), and [ADR 20261124](20261124-stack-core-errors-and-fallible-companion-apis.md))

## Context

Following the introduction of Tier 0 `math_diagnostics` (`#1841`), Tier 0 `math_rng` (`#1844`), optional `#![no_std]` `serde` (`#1852`), and stack-allocated core errors (`#1854`), four error-hierarchy and portability gaps remained across the workspace:

1. **Tier 0 `math_rng` error diagnostics and umbrella propagation:** `QmcError` (`crates/00_foundation/math_rng/src/qmc/error.rs`) and `RngError` (`crates/00_foundation/math_rng/src/scalar.rs`) lacked `math_diagnostics::Diagnostic` implementations (`RNG-QMC-001..004`, `RNG-001`), were not re-exported by `math_commons::error` or `math_explorer::error`, and could not propagate with `?` into `MonteCarloError`, `PureMathError`, the six domain crate umbrellas, or `math_explorer::error::Error`.
2. **Alloc-gated `#![no_std]` symbolic AST:** `math_commons::ast` required `feature = "std"` for lexing, parsing, stack-machine compilation (`CompiledExpr`), GLSL/WGSL shader emission, and AST error types (`LexError`, `ParseError`, `EvalError`, `GlslError`, `ShaderEmitError`), blocking `no_std + alloc` and `no_std + serde` callers from using symbolic formulas while preserving `#1854`'s zero-`alloc` default-disabled `math_commons` baseline.
3. **Causal `KellyError` linear algebra retention:** `KellyError::NotPositiveDefinite` and `pure_math::statistics::kelly::portfolio` discarded non-SPD `engine_linalg::LinalgError` causes (`SingularMatrix`, `DimensionMismatch`, `WorkspaceTooSmall`) instead of preserving causal `std::error::Error::source()` chains.
4. **Domain umbrella extensibility, GUI diagnostic emission, and Tier 0 `api_drift` enforcement:** `AIError` and `ClimateError` lacked `#[non_exhaustive]` and `MathCommons(#[from] MathCommonsError)` variants; `CeraTool` (`cera.rs`) and `compile_custom_fit_expr` (`fit_bridge.rs`) converted domain/AST errors to strings without emitting structured diagnostics onto `federated_registry`; and `unified_verification::api_drift` did not enforce Tier 0 `math_rng` error re-exports or scan child `mod conversions;` submodules.

## Decision

1. **Tier 0 `QmcError` & `RngError` Diagnostics and Full Umbrella Propagation:**
   - Implement zero-allocation `math_diagnostics::Diagnostic` for `QmcError` (`RNG-QMC-001..004`, plus `Hash`) and `RngError` (`RNG-001`).
   - Re-export `QmcError` and `RngError` unconditionally in `math_commons::error`, `math_explorer::error`, and `math_explorer::prelude`, with `From<QmcError>` and `From<RngError>` across `MathCommonsError`, `MonteCarloError`, `PureMathError`, all six domain crate umbrellas (`AIError`, `AppliedError`, `BiologyError`, `ClimateError`, `EpidemiologyDomainError`, `PhysicsError`), and `math_explorer::error::Error`.
   - Expose `emit_to_global_bus`, `try_emit_with_source`, and `emit_with_source` on `math_explorer::error::Error` to satisfy `unified_verification::error_contracts`.

2. **Two-Tier `#![no_std]` Contract in `math_commons` (`--no-default-features` vs `--features alloc`):**
   - Preserve bare `math_commons --no-default-features` as strictly zero-`alloc` (`MathError`, `FitError`, `LinalgError`, `QmcError`, `RngError`, and core `MathCommonsError`).
   - Gate `pub mod ast` and `MathCommonsError::{Lex, Parse, Eval, Glsl, ShaderEmit}` on `#[cfg(feature = "alloc")]`, porting `lexer`, `parser`, `expr`, `eval`, `compiled`, `glsl`, and `shader` to `core` + `alloc` + `libm::atan2`, while keeping `HashMap`-backed `Expr::eval*`, `codegen`, and `latex` helpers under `#[cfg(feature = "std")]`.

3. **Causal `KellyError` Linear Algebra Chaining:**
   - Add `KellyError::Linalg(engine_linalg::LinalgError)` and `From<LinalgError> for KellyError` (mapping `NotPositiveDefinite` to `KellyError::NotPositiveDefinite` and retaining `Some(&LinalgError::NotPositiveDefinite)` from `Error::source()`), and delegate inner `DiagnosticVisitor` fields in `pure_math`.

4. **Domain Umbrellas, GUI Telemetry, and Tier 0 `api_drift` Guards:**
   - Mark `AIError` and `ClimateError` `#[non_exhaustive]`, add `MathCommons(#[from] MathCommonsError)`, and extract modular `error/conversions.rs` submodules.
   - Emit structured diagnostics via `crate::telemetry::emit_diagnostic` in `CeraTool::train_epochs` (`"CeraEmulator"`) and `compile_custom_fit_expr` (`"CustomFitBridge"`).
   - Extend `unified_verification::api_drift` to audit `TIER0_FOUNDATION_PREFIXES` (`math_diagnostics`, `math_rng`), ignore sealed `private::*` supertraits, and traverse non-inline child `mod` files when verifying umbrella `From<E>` implementations.

## Consequences

- Every Tier 0 `math_rng` and Tier 2 engine error converts in one `?` hop into any crate umbrella or `math_explorer::error::Error` with zero heap allocation during `Diagnostic::visit_fields`.
- Embedded and WebAssembly callers can choose bare stack-only `math_commons` (`--no-default-features`) or symbolic `CompiledExpr`/shader compilation with `--no-default-features --features alloc` (and `serde`).
- Any future Tier 0 foundation error omitted from `math_explorer::error` or a crate umbrella fails `xtask check-api-drift` immediately.
