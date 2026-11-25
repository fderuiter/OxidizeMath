# ADR 20261124: `#![no_std]` Scalar Foundation (`math_scalars`), Forward-Mode AD, and Outward-Rounded Enclosures

## Status

Accepted

## Context

Across OxidizeMath's differential-equation engines (`engine_diffeq`), curve fitters (`math_commons::fit`), differential-geometry and Riemannian tensor modules (`pure_math`), root solvers (`pure_math::analysis::roots`), and cortical surface reconstruction kernels (`domain_applied::applied::freesurfer`), numerical derivatives previously relied on fixed-step central finite differences (`h = 1e-5` to `1e-7`, or voxel-scale steps). Fixed finite-difference stencils suffered from three structural limitations:

1. **Precision Loss and Cancellation**: In `f32` at moderate state magnitudes (such as `x = 64.0_f32` in Milstein SDE diffusion derivatives), fixed `1e-5` steps rounded to coarse binary fractions and lost multiple digits of precision; nested second-order finite differences in `ParametricSurface` Gaussian curvature, `laplace_beltrami`, curvilinear `laplacian`, and FreeSurfer `normal_external_force` amplified subtractive cancellation.
2. **Large-Scale Stagnation and Uncertified Root Bracketing**: Fixed `h = 1e-7` steps underflowed `x + h == x` at `x = 3.0e9` in `NewtonRaphson`, and scalar floating-point root finders could not provide rigorous outward-rounded root containment guarantees.
3. **Sub-Machine-Epsilon `f32` Tolerances**: `TOLERANCE_HIGH_F32` (`1e-7`) and `TOLERANCE_STANDARD_F32` (`1e-7`) in `math_commons::registry` were strictly smaller than `f32::EPSILON` (`1.1920929e-7`), making exact convergence criteria unachievable in unit-scale `f32` arithmetic.

Furthermore, WebAssembly (`wasm32-unknown-unknown`) lacks portable hardware FPU rounding-mode control (`fesetround`), requiring any interval or ball arithmetic to achieve outward rounding purely in IEEE-754 bit-level and error-free software transformations.

## Decision

1. **Introduce Tier 0 `crates/00_foundation/math_scalars`**:
   - `#![no_std]` with optional `alloc`, `serde` (with domain-invariant validation on `Interval<T>` and `Ball<T>` and const-generic array serialization on `DualVec<T, const N: usize>`), and `std` features, depending exclusively on `libm` (`0.2`) and optional `#![no_std]` `serde` with zero upward workspace crates.
   - Separate `Scalar` (differentiable and enclosure algebraic/transcendental operations implemented for `f32`, `f64`, `Dual<T>`, `DualVec<T, const N: usize>`, `HyperDual<T>`, `Interval<T>`, and `Ball<T>`) from `RealScalar` (ordered IEEE-754 floating-point operations and ULP stepping for `f32` and `f64`), while preserving compatibility for existing `math_commons::math_kernel::types::Scalar` callers.
   - Implement portable outward-rounded `Interval<T>` (`[lo, hi]`) and midpoint-radius `Ball<T>` (`mid ± rad`) using bit-level `next_up`/`next_down`, Knuth TwoSum error-free addition/subtraction bounds, `libm::fma` residual product/quotient/sqrt bounds, quadrant-aware trigonometric extrema, and tight ULP-padded `libm` elementary enclosures without calling `fesetround`.
2. **Propagate Zero-Allocation Forward-Mode AD Across Engines and Domains**:
   - `engine_diffeq`: Add `DifferentiableSde` and `AdSde<D>` to evaluate exact `Dual<T>` Milstein diffusion derivatives (within `<= 1` ULP in `f32` at `x = 64.0_f32`), while upgrading legacy finite-difference fallbacks to scale-aware `sqrt(EPSILON)` steps.
   - `math_commons`: Add generic `CompiledExpr::eval_*_scalar` evaluators, single-pass `DualVec<f64, MAX_FIT_PARAMS>` Levenberg-Marquardt solvers (`levenberg_marquardt_ad`), `sqrt(EPSILON)` fallback Jacobians, and recalibrated `f32` registry tolerances (`TOLERANCE_HIGH_F32 = 2.5e-7`, `TOLERANCE_STANDARD_F32 = 5.0e-7`, `TOLERANCE_FAST_F32 = 1.0e-6` — all strictly above `f32::EPSILON`).
   - `pure_math`: Upgrade `ParametricSurface` (`SurfaceJet2`, `Torus`, `Sphere`, `KleinBottle`, compiled AST surfaces, `laplace_beltrami_ad`), curvilinear `gradient_ad`/`divergence_ad`/`curl_ad`/`laplacian_ad`, `AdRiemannianMetric` / `AdStaticRiemannianMetric` Christoffel & covariant derivatives, scale-aware + `Dual<f64>` `NewtonRaphson` (converging cleanly at `3.0e9`), and certified `IntervalNewton` and `Krawczyk` solvers enclosing `sqrt(2)` and transcendental roots within `<= 4` ULPs.
   - `domain_applied`: Upgrade FreeSurfer `MriVolume` to a $C^\infty$ separable Gaussian filter differentiated via single-pass `DualVec<f64, 3>` (`smoothed_gradient`) and 3-pass `HyperDual<f64>` (`normal_external_force_ad`) to `< 1e-13` accuracy with zero heap allocations.
3. **Record Certified Enclosure Telemetry in `Verification_Certificate.json`**:
   - Extend `unified_verification` and `Verification_Certificate.json` (mirrored in `oxidize_core::traceability::VerificationCertificate` and `.github/workflows/ci.yml`) with `certified_interval_max_width`, `certified_root_max_ulp_bound` (`<= 4` ULPs), and `certified_enclosures_verified`.

## Consequences

- **Positive**: Exact, zero-allocation first- and second-order derivatives (`< 1e-13` in `f64`, `<= 1` ULP in `f32`) eliminate step-size tuning and subtractive cancellation across SDEs, nonlinear least squares, differential geometry, general relativity, root finding, and cortical MRI reconstruction.
- **Positive**: `Interval<T>` and `Ball<T>` produce deterministic, identical certified enclosures (`<= 4` ULPs) on both `x86_64` and `wasm32-unknown-unknown` without mutating hardware FPU flags.
- **Trade-offs**: Workspace crate count increases from 25 to 26 crates, and `Verification_Certificate.json` schema expands from 9 to 12 deterministic keys. Reverse-mode tape autodiff for high-dimensional neural parameter tensors remains out of scope for stack-allocated forward-mode scalars.
