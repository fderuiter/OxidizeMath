# ADR 20261124: Tier 0 `math_scalars` Crate, Zero-Allocation Forward AD, and Certified IEEE-754 Enclosures

## Status
Accepted (2026-11-24)

## Context

Prior to this decision, solvers across `math_commons`, `engine_diffeq`, `pure_math`, and `domain_applied` exhibited five coupled numerical and architectural limitations:

1. **Order-coupled scalar bounds:** `RealScalar` required `PartialOrd`, preventing differentiable (`Dual<T>`, `DualVec<T, N>`, `HyperDual<T>`) and set-valued enclosure (`Interval<T>`, `Ball<T>`) numbers from instantiating generic kernels, AST evaluators, and solver workspaces.
2. **Finite-difference cancellation and scale collapse:** Fixed step sizes (`1e-5` to `1e-7`) across SDE Milstein/SRIW1 diffusion Jacobians, Levenberg-Marquardt curve fitting, differential-geometry curvature and Laplace-Beltrami operators, curvilinear vector calculus, Riemannian Christoffel symbols, FreeSurfer cortical surface forces, and `NewtonRaphson` suffered subtractive cancellation on second derivatives and collapsed to zero when $|x| \gg 1$ (e.g., `NewtonRaphson` failing at $3\times 10^9$ and `f32` SDE diffusion derivatives drifting by tens of thousands of ULPs at $x = 64.0_{\text{f32}}$).
3. **`f32` tolerance inversion:** `crates/math_commons/src/registry.rs` set `TOLERANCE_HIGH_F32 = 1e-7 < f32::EPSILON` (`1.1920929e-7`), creating unreachable single-precision convergence thresholds.
4. **Non-portable FPU rounding modes:** `wasm32-unknown-unknown` lacks hardware `fesetround` directed rounding modes, requiring portable bit-level IEEE-754 error-free transformations and conservative `libm` ULP padding to guarantee identical outward-rounded intervals on desktop and WebAssembly.
5. **Uncertified root enclosures:** Root finders returned only point estimates without verified Interval-Newton or Krawczyk existence/uniqueness brackets recorded in `Verification_Certificate.json`.

## Decision

1. **Create Tier 0 `#![no_std]` crate `crates/00_foundation/math_scalars`:**
   - Depend solely on deterministic `libm` (`default-features = false`) with optional `alloc` support and zero upward workspace dependencies on both native (`x86_64`) and `wasm32-unknown-unknown`.
   - Split `Scalar` (field arithmetic plus deterministic `libm` elementary functions implemented uniformly across `f32`, `f64`, `Dual<T>`, `DualVec<T, N>`, `HyperDual<T>`, `Interval<T>`, and `Ball<T>`) from `RealScalar: Scalar + PartialOrd + DirectedRounding`.
   - Provide zero-allocation first-order (`Dual<T>`), multi-tangent (`DualVec<T, const N: usize>`), and second-order (`HyperDual<T>`) forward-mode automatic differentiation.
   - Provide outward-rounded `Interval<T>` and midpoint-radius `Ball<T>` enclosures via error-free `TwoSum` / Dekker-Veltkamp `TwoProd` transformations, IEEE-754 bit-level `next_down` / `next_up` ULP stepping, stationary-point trigonometric range bounds, and certified `interval_newton_enclose`, `krawczyk_enclose`, and `certify_reference_roots` routines without `fesetround`.

2. **Upgrade `math_commons` and `engine_diffeq`:**
   - Raise `f32` tolerances in `math_commons::registry` strictly above `f32::EPSILON` (`TOLERANCE_HIGH_F32 = 2.5e-7`, `TOLERANCE_STANDARD_F32 = 5.0e-7`, `TOLERANCE_FAST_F32 = 1.0e-6`), relax `SolverWorkspace<'a, T: Copy>`, add generic `CompiledExpr::eval_*_scalar` bytecode evaluation, and add single-pass `DualVec<f64, MAX_FIT_PARAMS>` `levenberg_marquardt_ad` curve fitting while upgrading legacy finite-difference fallbacks to scale-aware $\sqrt{\varepsilon}\max(1, |\theta_j|)$ steps.
   - Replace finite-difference diffusion derivatives in `engine_diffeq` (`SdeSystem`, `MilsteinSolver`, `Sriw1Solver`) with exact zero-allocation `Dual<T>` directional derivatives via `SdeSystem<Dual<T>>`.

3. **Upgrade `pure_math`, `domain_applied`, and `unified_verification`:**
   - Introduce `DifferentiableSurface` (`HyperDual<f64>` exact `Sphere`, `Torus`, `KleinBottle`, `CompiledParametricSurface`, `CompiledHeightfieldSurface`, and `laplace_beltrami_ad`), `DifferentiableCoordinates` (`gradient_ad`, `divergence_ad`, `curl_ad`, `laplacian_ad`), `AdRiemannianMetric` / `StaticAdMetric` / `covariant_derivative_*_ad`, $C^2$ cubic B-spline + `HyperDual<f64>` FreeSurfer `normal_external_force`, and multi-tangent AD battery degradation calibration.
   - Add `NewtonRaphson::find_root_ad`, scale-aware `NewtonRaphson::find_root`, `IntervalNewton`, and `Krawczyk` in `pure_math::analysis::roots`, and enforce `certified_interval_max_ulp_width <= 4`, `certified_ball_max_ulp_width <= 4`, and `certified_roots_contained == true` in `unified_verification` and `Verification_Certificate.json`.

## Consequences

- **Positive:** Eliminates subtractive cancellation and large-magnitude step underflow across SDE solvers (`<= 1` ULP in `f32` at $x = 64$), nonlinear least squares, differential geometry, vector calculus, tensor calculus, cortical surface deformation (`< 1e-13` in `f64`), and root finding with zero dynamic heap allocations under `VerifiedEngine`.
- **Positive:** Provides deterministic, FPU-mode-free outward-rounded `Interval<T>` and `Ball<T>` enclosures (`<= 4` ULPs on both `x86_64` and `wasm32-unknown-unknown`) verified by `unified_verification`.
- **Neutral:** Workspace member count advances from `24` to `25` crates; reverse-mode tape autodiff for large parameter tensors remains out of scope.
