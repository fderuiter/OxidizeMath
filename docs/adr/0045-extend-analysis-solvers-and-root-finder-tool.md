# ADR 0045: Extend Analysis Solvers and Standalone Root Finder Tool

## Status

Accepted (2026-11-23)

## Context

Previously, `pure_math::analysis` had several algorithmic and ergonomic gaps across its numerical analysis kernels and GUI workbenches:

1. `TimeStepper` and `OdeModel` always passed `t = 0.0` to `OdeSolver::step`, preventing `OdeModel` from integrating non-autonomous systems $y'(t) = f(t, y)$, while `pure_math::analysis::ode` lacked adaptive explicit (`DormandPrince45`), stiff implicit (`BackwardEuler`, `ImplicitTrapezoidal`), and linearly implicit Rosenbrock-Wanner (`Rosenbrock23`) solvers.
2. `Bisection` relied on `f64::signum()`, which treats `-0.0` as `-1.0` and silently accepts `NaN` or infinite interval evaluations, while `pure_math::analysis::roots` lacked `Brent`, `Secant`, and `MultivariateNewton`.
3. `Trapezoidal` returned `0.0` as its error estimate, and `pure_math::analysis::integration` lacked composite `Simpson` and stack-bounded adaptive `GaussKronrod15`.
4. `AnalysisTab` in `math_explorer_gui` had no standalone root-finding workbench and only exposed two quadrature rules and three explicit fixed-step ODE methods.

## Decision

1. **Zero-Allocation Dense Linear & Sign Helpers (`pure_math::analysis::dense_linear`):**
   - Added internal stack-friendly helpers (`canonical_zero`, `has_strict_sign_change`, `solve_linear_in_place`, `lu_factor_in_place`, `lu_solve_in_place`) operating on caller-supplied slices without heap allocation.

2. **Extended Root-Finding (`pure_math::analysis::roots`):**
   - Hardened [`bracketing.rs`](../../crates/pure_math/src/pure_math/analysis/roots/bracketing.rs) so `Bisection` rejects `NaN`/`Inf` endpoints and evaluations with `AnalysisError::InvalidParameters`, canonicalizes `-0.0` exact roots to `0.0`, and tests strict sign changes without `f64::signum()`. Added Brent-Dekker `Brent`.
   - Added derivative-free `Secant` alongside hardened `NewtonRaphson` in [`open.rs`](../../crates/pure_math/src/pure_math/analysis/roots/open.rs), and added stack-allocated (`SVector<f64, N>`) and pre-allocated dynamic (`DVector<f64>`) `MultivariateNewton` in [`multivariate.rs`](../../crates/pure_math/src/pure_math/analysis/roots/multivariate.rs).

3. **Extended Quadrature (`pure_math::analysis::integration`):**
   - Upgraded `Trapezoidal` in [`mod.rs`](../../crates/pure_math/src/pure_math/analysis/integration/mod.rs) with a two-grid Richardson + kink-safe slope-variation error bound, added composite `Simpson`, and added stack-bounded 64-panel adaptive `GaussKronrod15` (`G7-K15`) with panel inspection (`integrate_panels`) in [`gauss_kronrod.rs`](../../crates/pure_math/src/pure_math/analysis/integration/gauss_kronrod.rs).

4. **Time-Aware, Adaptive, and Stiff ODE Solvers (`pure_math::analysis::ode`):**
   - Added defaulted `current_time` and `advance_time` methods to [`stepper.rs`](../../crates/pure_math/src/pure_math/analysis/ode/stepper.rs) and `pub time: f64` tracking to [`model.rs`](../../crates/pure_math/src/pure_math/analysis/ode/model.rs).
   - Added additive `IndexedState` in [`traits.rs`](../../crates/pure_math/src/pure_math/analysis/ode/traits.rs) and [`state.rs`](../../crates/pure_math/src/pure_math/analysis/ode/state.rs) (`VecState`, `ArrayState<N>`, `Vector2<f64>`, `Vector3<f64>`, `DVector<f64>`).
   - Added `DormandPrince45`, `BackwardEuler`, `ImplicitTrapezoidal`, and `Rosenbrock23`, expanding `IntegrationMethod::ALL` to 7 solvers while preserving zero heap allocations on stack states under `VerifiedAllocator`.

5. **Interactive Analysis Workbenches (`math_explorer_gui/src/tabs/analysis/`):**
   - Registered standalone [`root_finder.rs`](../../math_explorer_gui/src/tabs/analysis/root_finder.rs) (`RootFinderTool`) in [`mod.rs`](../../math_explorer_gui/src/tabs/analysis/mod.rs) supporting 1D (`Bisection`, `Brent`, `NewtonRaphson`, `Secant`) brackets/chords/tangents and 2D `MultivariateNewton` zero-contour overlays, copyable residual telemetry, starter presets, AccessKit labels, and deferred plot exports.
   - Upgraded [`riemann.rs`](../../math_explorer_gui/src/tabs/analysis/riemann.rs) with `Simpson` parabolic arcs and `GaussKronrod15` adaptive panel visualization, and upgraded [`ode.rs`](../../math_explorer_gui/src/tabs/analysis/ode.rs) with all 7 `IntegrationMethod` solvers, adaptive tolerance controls, stiff decay preset, and multi-solver comparative overlays.

## Consequences

- Fixed-size root-finding (`Bisection`, `Brent`, `NewtonRaphson`, `Secant`, `MultivariateNewton<2>`), quadrature (`Trapezoidal`, `Simpson`, `GaussKronrod15`), and ODE stepping (`DormandPrince45`, `BackwardEuler`, `ImplicitTrapezoidal`, `Rosenbrock23` on `ArrayState<N>`) execute with zero dynamic heap allocations under `VerifiedAllocator`.
- `OdeModel` integrates non-autonomous systems accurately (`y'(t) = t` within `1e-12`), and implicit solvers keep stiff decay (`y' = -1000 * y` at `dt = 0.05`) strictly positive and bounded without oscillation.
- All existing public traits and signatures in `pure_math` remain backward-compatible.
