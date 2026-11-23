# 0037. Hierarchical Umbrella Error Enums, Full Submodule Error Re-Exports, and DiagnosticBus Emission Helpers

## Status

Accepted

* **Date:** 2026-11-22

## Context

`math_explorer` serves as the unified public SDK facade for OxidizeMath (see [ADR 0034](./0034-curate-math-explorer-facade-exports-and-doctests.md)). While crate-root `src/error.rs` files exposed 26 error enums, 22 public error types defined in nested submodules (such as `KalmanError`, `ClinicalTrialError`, `VoxelGridError`, `LinearAlgebraError`, `AnalysisError`, `EvolutionError`, `OptimizationError`, `StochasticError`, `TensorError`, `TransformError`, `EllipticCurveError`, `AmbsError`, `KineticsError`, `ReactionDiffusionError`, `NetworkModelError`, `GatingError`, `NuclearError`, `EvalError`, `GlslError`, `LexError`, and `ParseError`) were omitted from `math_explorer::error` because `crates/unified_verification/src/api_drift/facade.rs` inspected only `crates/*/src/error.rs`. In addition, `crates/pure_math/src/pure_math/number_theory/ualbf.rs` defined a duplicate `NumberTheoryError` that collided with `crates/pure_math/src/error.rs::NumberTheoryError`.

Furthermore, callers composing sibling submodules within a domain crate or multi-domain workflows through `math_explorer` lacked crate-level and facade-level umbrella error enums. Combined with 11 wrapper error types that dropped their inner `std::error::Error::source()` cause, 19 submodule errors missing `diagnostics::Diagnostic`, and `DiagnosticBus` lacking a helper to convert `&dyn Diagnostic` directly into `ModuleDiagnostic` events (after dead channel helpers were pruned in [ADR 0027](./0027-prune-diagnostic-bus-channel-and-dead-helpers.md)), contributors were forced into manual `.unwrap()` calls or heap-allocated `Box<dyn std::error::Error>` workarounds that violate zero-allocation runtime rules ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

## Decision

1. **Deduplicate `NumberTheoryError` and Re-Export All 47 Leaf Errors**:
   - Consolidate `crates/pure_math/src/pure_math/number_theory/ualbf.rs` onto `crates/pure_math/src/error.rs::NumberTheoryError`.
   - Re-export all 21 previously omitted submodule error types through their parent modules, their crate `src/error.rs` modules (`<domain>::error`), `math_explorer::error`, and `math_explorer::prelude` under their respective Cargo feature gates.

2. **Introduce Extensible Crate-Level Umbrella Error Enums**:
   - Add `#[non_exhaustive]` crate umbrella enums in all 6 multi-error crates:
     - `math_commons::error::MathCommonsError`
     - `pure_math::error::PureMathError`
     - `domain_applied::error::AppliedError`
     - `domain_biology::error::BiologyError`
     - `domain_epidemiology::error::EpidemiologyDomainError`
     - `domain_physics::error::PhysicsError`
   - Implement zero-allocation `From` conversions from every sibling leaf error in the crate as well as foundational `MathError` / `MathCommonsError` and coupled upstream domain errors.

3. **Provide Feature-Gated `math_explorer::error::Error` and `Result`**:
   - Define `#[non_exhaustive] pub enum Error` and `pub type Result<T, E = Error> = std::result::Result<T, E>;` in `math_explorer::error` (re-exported in `math_explorer::prelude`), gating domain variants behind `pure_math`, `ai`, `applied`, `biology`, `epidemiology`, and `physics` so `default = []` remains strictly limited to `math_commons` and `diagnostics`.
   - Provide one-hop `From` conversions for all 47 leaf error types and all 6 crate umbrella enums without intermediate boxing or heap allocation.

4. **Preserve Causal `Error::source()` Chains and Uniform `Diagnostic` Telemetry**:
   - Implement `std::error::Error::source()` across all 15 leaf wrapper error types and 100% of crate and facade umbrella variants, avoiding `#[error(transparent)]` where inner errors have `None` sources so umbrella `.source()` always yields the wrapped error.
   - Implement `diagnostics::Diagnostic` across all submodule errors, all 6 crate umbrella enums, and `math_explorer::error::Error`.
   - Add `DiagnosticBus::emit_diagnostic` and `DiagnosticBus::emit_diagnostic_with_source` in `crates/diagnostics/src/lib.rs` so callers can publish any `&D` (`D: Diagnostic + ?Sized`) directly to registered listeners such as `federated_registry`.

5. **Enforce Submodule Error Completeness and Uniqueness in `unified_verification`**:
   - Extend `crates/unified_verification/src/api_drift/facade.rs` (via `facade_errors.rs`) to audit all facade-reachable submodules in extracted API maps, failing CI if any reachable public error `struct` or `enum` is missing from `math_explorer/src/error.rs` or if duplicate error names exist across reachable modules.

## Consequences

* **Positive:** SDK consumers and workspace contributors can compose intra-crate and cross-domain operations with `?` using `math_explorer::error::Result` or crate umbrella enums with zero heap allocations on both `Ok` and `Err` paths under `VerifiedEngine::run_verified`.
* **Positive:** Runtime causal chains (`Error::source()`) and structured `Diagnostic` metadata are preserved end-to-end across all 47 leaf errors and umbrella wrappers, and `api_drift` blocks future facade error omissions or naming collisions automatically.
* **Trade-offs:** Adding new public error types in any reachable module requires re-exporting them in `math_explorer/src/error.rs` and regenerating `crates/unified_verification/api_baseline.json`.
