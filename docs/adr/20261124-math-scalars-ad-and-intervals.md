# ADR 20261124: `#![no_std]` Forward-Mode Automatic Differentiation, IEEE-754 Interval/Ball Enclosures (`math_scalars`), and Tier 5 Root Certification

## Status

Accepted

## Context

Prior to this decision, derivative-dependent algorithms and root finders across `engine_diffeq`, `math_commons`, `pure_math`, and `domain_applied` relied on step-tuned finite-difference stencils ($O(h^2)$) with several structural failure modes:

1. **Step-size truncation vs. cancellation trade-offs**:
   - `SdeSystem::diffusion_jacobian_action` in `engine_diffeq` used a fixed `1e-6` perturbation, which underflowed completely in `f32` whenever $|x_i| > 32$ and added truncation error to Milstein and Stratonovich Wong-Zakai drift corrections.
   - `pure_math::tensor` Christoffel symbols and numerical differentiation used `h = 1e-5..1e-6`, amplifying $O(\epsilon / h^2)$ cancellation noise near coordinate horizons.
   - `pure_math::vector_calculus::operators` (`gradient`, `divergence`, `curl`, `laplacian`), `differential_geometry::surface` (first and second fundamental forms, Gaussian and mean curvature), and `domain_applied::freesurfer` cortical surface forces used multi-pass finite differences ($2k + 1$ callbacks in Levenberg-Marquardt curve fitting).
2. **Uncertified root-finding brackets**:
   - `pure_math::analysis::roots` exposed floating-point bisection, Newton-Raphson, and secant solvers without a Brent-Dekker solver or mathematically certified outward-rounded IEEE-754 interval/ball root enclosures.
3. **Rigid workspace scratch trait bounds**:
   - `SolverWorkspace<'a, T: Scalar>` restricted zero-allocation scratch partitioning to `f32`/`f64`, preventing reuse with dual jets or interval slices.
4. **Upstream `#![no_std]` `serde`, stack core errors, and 6-tier verification pyramid alignment**:
   - Following the `#![no_std]` optional `serde` architecture (`#1827`), bounded `math_commons::primitives::Interval` (`#1844`), checked companion APIs and stack-allocated `MathError` / `DiffeqError` + AST error contract gates (`#1852`, `#1854`), and the 6-tier scientific verification pyramid (`#1841`, `#1844`), the new Tier 0 scalar foundation must preserve optional `#![no_std]` `serde` support, explicit `alloc` feature gating, `try_*` fallible companions (`SdeSystem::try_diffusion_jacobian_action`, `MilsteinSolver::try_step`, `Sriw1Solver::try_step`, `try_eval_ito_effective_drift`), bridge cleanly to `math_commons::primitives::Interval`, and plug root-enclosure ULP certification directly into Tier 5 (`scientific_verification`) without mutating the 14-key `Verification_Certificate.json` schema.

## Decision

1. **Create Tier 0 `#![no_std]` crate `crates/00_foundation/math_scalars`** (27th workspace crate, 16th release crate):
   - Define algebraic `Scalar` (`f64`, `f32`, `Dual<T>`, `DualVec<T, N>`, `HyperDual<T>`, `Interval<T>`, `Ball<T>`) and `RealScalar: Scalar + DirectedRounding` (`f64`, `f32` with `next_up`, `next_down`, `ulp_distance`, and directed `+`, `-`, `*`, `/`, `sqrt` via `libm`).
   - Implement stack-allocated forward-mode automatic differentiation types: `Dual<T>` (first-order directional tangent), `DualVec<T, const N: usize>` (single-pass multi-tangent gradient/Jacobian row), and `HyperDual<T>` ($a + b\varepsilon_1 + c\varepsilon_2 + d\varepsilon_1\varepsilon_2$, exact truncation-free second and mixed partial derivatives).
   - Implement outward-rounded IEEE-754 `Interval<T>` (`[lo, hi]`) and midpoint-radius `Ball<T>` (`[mid - rad, mid + rad]`) arithmetic with exact elementary range enclosures (`sqr`, `powi`, `sqrt`, `exp`, `ln`, `sin`, `cos`, `tan`, `atan`), plus `interval_newton_enclose`, `krawczyk_enclose`, and `certify_reference_roots`.
   - Provide optional `#![no_std]` `serde` (`serde = ["dep:serde"]`, `std = ["alloc", "serde?/std"]`) in `src/serde_impls.rs` enforcing invariant validation (`lo <= hi` on `Interval<T>`, `rad >= 0` on `Ball<T>`, and const-generic length validation on `DualVec<T, N>`).
2. **Relax `SolverWorkspace` and bridge `math_commons::primitives::Interval`**:
   - Relax `SolverWorkspace<'a, T>` to `T: Copy` (preserving `try_take_slice` and `Default`-backed `zero_fill`), add `CompiledExpr::eval_{slice,1d,2d,3d}_scalar<S: Scalar>` and single-pass `levenberg_marquardt_ad` (`DualVec<f64, 6>`), fix `TOLERANCE_HIGH_F32` / `TOLERANCE_STANDARD_F32` above `f32::EPSILON`, and bridge `math_commons::primitives::Interval` <-> `math_scalars::Interval<f64>` via `From`/`TryFrom`.
3. **Upgrade SDE, Geometry, Tensor, Root-Finding, and Applied Solvers to Exact Forward AD & Certified Enclosures**:
   - Upgrade `engine_diffeq` (`SdeSystem::diffusion_jacobian_action`, `MilsteinSolver`, `Sriw1Solver`) to stack-allocated `Dual<T>`, eliminating `f32` finite-difference step underflow while preserving `#![no_std]` `serde` and `BaoabLangevinSolver::try_new`.
   - Add `brent`, `enclose_root_interval`, and `enclose_root_ball` in `pure_math::analysis::roots`; `StaticMetric<N>` exact `DualVec` Christoffel symbols and `Dual`/`HyperDual` differentiation in `pure_math::tensor`; exact `DualVec`/`HyperDual` operators in `vector_calculus` and `differential_geometry` (including `CompiledParametricSurface`); and `DualVec`/`HyperDual` cortical MRI gradients/forces and battery aging calibration in `domain_applied`.
4. **Certify Reference Root Enclosures in Tier 5 of `scientific_verification`**:
   - Register `root_sqrt2_{interval,ball}`, `root_ln3_{interval,ball}`, and `root_dottie_{interval,ball}` inside `scientific_verification::intervals::compute_certified_interval_enclosures`, enforcing containment and `<= 4` ULP width across `verify-tier5`, `verify-pyramid`, `check-certificate`, and `lint-all`.

## Consequences

- Forward-mode derivatives are exact to machine precision (`<= 2` ULPs) with zero post-initialization heap allocations under `VerifiedEngine`.
- `f32` Milstein and Stratonovich diffusion Jacobians remain accurate at large state magnitudes (`|x| = 64_f32` within `1` ULP).
- Reference algebraic and transcendental roots (`sqrt(2)`, `ln(3)`, Dottie fixed point) are mathematically enclosed within `<= 4` ULPs in Tier 5 without altering the 14-key `Verification_Certificate.json` schema.

```rust
use math_scalars::{DualVec, Interval, Scalar};

let seed = DualVec::<f64, 2>::jacobian_seed([1.0, 2.0]);
let f = (seed[0] * seed[1]).exp();
assert!((f.eps[0] - 2.0 * 2.0_f64.exp()).abs() < 1e-14);

let tenth = Interval::point(1.0_f64) / Interval::point(10.0_f64);
assert!(tenth.contains(0.1));
```

