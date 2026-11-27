# ADR 20261126: Extract Tier 0 `math_scalars` Hierarchy and Tier 2 `engine_opt` Curve-Fitting Engine

## Status

Accepted

## Context

Prior to this decision, algebraic and numerical scalar abstractions were split across four disconnected workspace layers while Tier 2 `math_commons` carried a direct dependency on `engine_linalg`:

1. **Fragmented Scalar Traits and IEEE-754 Kernels**: `pure_math::algebra::traits::{Ring, Field}`, `engine_linalg::Scalar`, `math_commons::math_kernel::types::Scalar`, and `math_rng::scalar::RngScalar` defined overlapping scalar contracts. Furthermore, `<f64 as engine_linalg::Scalar>::sqrt` used a fixed 5-step Heron iteration that exhibited 1-ULP rounding drift near dyadic boundaries (such as `3.9999999999999996_f64`) and overflow/underflow hazards on subnormal IEEE-754 inputs.
2. **Tier 2 Engine Coupling (`math_commons -> engine_linalg`)**: Curve-fitting solvers (`math_commons::fit`) forced `math_commons` to depend on `engine_linalg`, coupling lightweight symbolic AST, physical constants, and point-of-interest helpers to dense linear algebra factorizations.
3. **Hardcoded `f64` Solvers**: Core linear algebra (`engine_linalg`), cortical general linear modeling (`domain_applied::applied::freesurfer::glm`), and discrete sandpile Kirchhoff solvers (`domain_physics::physics::stat_mech::sandpile`) could not interoperate cleanly across `f32`, `f64`, forward-mode `Dual<T>` automatic differentiation, complex numbers, and arbitrary-precision `dashu` domains without a shared zero-allocation reference-operator hierarchy.

## Decision

1. **Extract Tier 0 `#![no_std]` [`math_scalars`](../../crates/00_foundation/math_scalars/README.md) Foundation Crate**:
   - Define the unified reference-aware algebraic and analytical trait hierarchy (`RefOps`, `Scalar`, `Ring`, `Field`, `ComplexField`, and `RealField`) supporting both `Copy` IEEE-754 scalars (`f32`, `f64`, `Complex<T>`, `Dual<T>`) and non-`Copy` heap domains (`pure_math::math_types::{Integer, Rational, Float}`, `Fp<P>`, `Polynomial<T>`, `LaurentPolynomial<T>`).
   - Back `sqrt_f32`, `sqrt_f64`, `hypot_f32`, and `hypot_f64` with `libm` (`0`-ULP IEEE-754 parity across normal and subnormal bit patterns) and relocate shared `#![no_std]` workspace and series primitives (`SolverWorkspace`, `StepSize`, `Dimension`, `GridIndex`, `Sample2D`, `PointSeries`, `ScalarError`).
   - Unify `math_rng::RngScalar` as a blanket supertrait extension over `math_scalars::RealField`.

2. **Extract Tier 2 `#![no_std]` [`engine_opt`](../../crates/02_engines/engine_opt/README.md) Curve-Fitting Engine**:
   - Relocate zero-allocation polynomial regression, nonlinear parameter seeding, and damped Levenberg-Marquardt curve-fitting solvers (`fit`, `FitModel`, `FitError`) from `math_commons` into `crates/02_engines/engine_opt`.
   - Remove the `engine_linalg` and `scientific_metadata` dependencies from [`math_commons`](../../crates/02_engines/math_commons/Cargo.toml), leaving `math_commons` independent of `engine_linalg`.

3. **Generalize Downstream Solvers and Synchronize Facades**:
   - Generalize `engine_linalg` (`lu`, `qr`, `cholesky`, `svd_*`, `krylov`, `tikhonov`, `norms`), `domain_applied::applied::freesurfer::glm::CorticalGlm<T>`, and `domain_physics::physics::stat_mech::sandpile::kirchhoff` over `math_scalars::RealField`.
   - Re-export `math_scalars`, `engine_opt`, and `engine_opt::fit` from [`oxidizemath`](../../crates/04_bindings/oxidizemath/src/lib.rs) (and transitively `math_explorer`), preserving causal `Diagnostic` and `Error::source()` chains across `FitError`, `LinalgError`, and `ScalarError`.

## Consequences

- **Positive**: Eliminates the Tier 2 `math_commons -> engine_linalg` dependency edge, guarantees `0`-ULP `libm` square-root parity across `f32` and `f64`, and enables generic `f32`/`f64`/`Dual<T>` linear algebra and GLM execution with zero heap allocations in inner loops.
- **Positive**: Unifies `Copy` and non-`Copy` ring and field abstractions under a single `#![no_std]` foundation crate without forcing `Clone` allocations inside polynomial or matrix kernels.
- **Trade-offs**: Workspace crate count increases from 28 to 30 crates, recorded in [`system_overview.md`](../architecture/system_overview.md), [`traceability.toml`](../../traceability.toml), [`verification_whitelist.toml`](../../verification_whitelist.toml), and [`api_baseline.json`](../../apps/unified_verification/api_baseline.json).
