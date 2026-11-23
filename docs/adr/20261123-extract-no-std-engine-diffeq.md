# ADR 20261123: Extract `#![no_std]` `engine_diffeq` for ODE Solvers, Quadrature, and Hermite Root Bracketing

## Status
Accepted (2026-11-23)

## Context

Previously, OxidizeMath's differential equation solvers (`Euler`, `Heun`, `RungeKutta4`, `OdeModel`, `TimeStepper`) and basic numerical quadrature (`Trapezoidal`, `ClenshawCurtis`) lived directly inside `pure_math`, which pulls in heavy `std`-only dependencies (`statrs`, `petgraph`, `rustfft`, `dashu`, `quadrature`). At the same time:

1. Embedded and `wasm32-unknown-unknown` consumers could not depend on ODE stepping, quadrature, or Hermite root bracketing without compiling `std` and heap-heavy dependencies.
2. Only explicit low-order Runge-Kutta solvers were available; non-autonomous adaptive error control (`DormandPrince45`) and stiff implicit solvers (`Rosenbrock23`, `Bdf2`) were missing.
3. `Trapezoidal::integrate` returned `error: 0.0` and ignored `eps`, forcing GUI tools (`RiemannIntegrationTool`) to run ad-hoc double evaluations.
4. Composite trapezoidal loops and safeguarded cubic Hermite Newton-bisection loops were duplicated across `domain_applied` (`pharmacokinetics`, `cannibalism`), `domain_physics` (`chaos::poincare`), and `math_commons` (`poi`).

## Decision

1. **Extract `crates/02_engines/engine_diffeq` (`#![no_std]`)**:
   - Compiles unconditionally under `#![no_std]` with `default = []` features (depending only on `verified_engine` with `default-features = false` and `serde` with `default-features = false`). Optional `alloc`, `std`, and `nalgebra` features unlock `VecState`, `Theory` metadata, and `nalgebra` vector implementations.
   - Provides `SolverWorkspace<'a>` and `StackWorkspace<CAP>` bump-slice arenas plus zero-allocation partial-pivoting LU factorization/solve for caller-supplied scratch memory.
2. **Elapsed-Time Tracking and Stiff/Adaptive ODE Suite**:
   - `TimeStepper` gains defaulted `get_time` / `set_time` methods (preserving existing autonomous domain implementations), and `OdeModel` tracks drift-free elapsed simulation time `t`.
   - Adds adaptive 7-stage `DormandPrince45` (5(4) FSAL with PI step-size control), L-stable 2-stage `Rosenbrock23` (finite-difference Jacobian + LU stage solves), and variable-step `Bdf2` (damped Newton-LU iteration with Rosenbrock23 startup).
3. **Quadrature and Safeguarded Hermite Consolidation**:
   - Enhances `Trapezoidal::integrate` with Richardson error estimation and iterative midpoint step doubling when `eps > 0.0`.
   - Adds composite `Simpson` and adaptive 15-point `GaussKronrod15` (`G7-K15`) backed by a fixed-capacity 64-subinterval stack priority buffer (`GK15_STACK_SUBINTERVALS`).
   - Consolidates duplicated trapezoidal and safeguarded cubic Hermite root loops across `math_commons::poi`, `domain_applied::pharmacokinetics`, `domain_applied::cannibalism`, `domain_physics::chaos::poincare`, and `math_explorer_gui`.
4. **Facade Re-Exports**:
   - Re-exports `engine_diffeq` as `math_explorer::diffeq`, `DiffeqError` via `math_explorer::error::DiffeqError`, and full backwards-compatible types under `math_explorer::pure_math::analysis::{ode, integration, roots}`.

## Consequences

- **Positive**: Zero heap allocations during ODE stepping, adaptive quadrature, and Hermite root finding (enforced by `VerifiedEngine::run_verified`), full `#![no_std]` / `wasm32-unknown-unknown` compatibility, stable stiff ODE integration at coarse `dt`, and single-source numerical kernels across crates.
- **Trade-offs**: Workspace member count increases from 20 to 21, and `verified_engine` gates `std::thread_local!` behind a default `std` feature flag.
