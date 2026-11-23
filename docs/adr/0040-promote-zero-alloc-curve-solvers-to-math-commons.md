# ADR-0040: Promote Zero-Allocation Curve and Critical-Point Solvers to `math_commons`

## Status

Accepted

## Context

Polynomial ordinary least squares (degrees 1–4), damped Levenberg-Marquardt nonlinear least squares, closed-form seed estimators, and sub-segment critical-point solvers (monotone cubic Hermite threshold crossings, vertical crossings, 3-point Lagrange parabolic extrema, and monotone/general 2D polyline intersections) were previously implemented inside `egui_plot/src/fit*` and `egui_plot/src/poi*` (Layer 5 UI).

Keeping foundational numerical solvers inside `egui_plot` created four architectural defects:

1. **Layer inversion & headless exclusion**: Domain crates (Layer 4) and default headless `math_explorer` callers (`default = []`) could not access curve fitting or sub-segment root/extremum detection without pulling ~158 GUI dependencies.
2. **Heap allocations & duplicated math in GUI callouts**: `math_explorer_gui` tabs either copied paired `&[f64]` trajectories into temporary `Vec<PlotPoint>` buffers (`ode/annotations.rs`, `capacity_fade.rs`) or re-implemented 3-point Lagrange parabolic vertex algebra inline (`sir/metrics.rs`).
3. **Degree clamping & silent non-convergence**: `FitType::Polynomial(usize)` clamped arbitrary degrees into `1..=4`, breaking `Eq`/`Hash` canonicality (`Polynomial(1)` vs `Linear`, `Polynomial(5)` vs `Polynomial(4)`), while `fit_nonlinear` fell back to unrefined seed parameters or `None` instead of surfacing typed diagnostics.
4. **Traceability blind spot**: Both `egui_plot` and `crates/math_commons` were listed in `IGNORED_TRACEABILITY_MEMBERS`, exempting curve solvers from `verified_engine` contract verification and zero-allocation checks.

## Decision

1. **Promote solvers into Layer 3 (`crates/math_commons`) and re-export in default `math_explorer`**:
   - Introduce `math_commons::series::{PointSeries, Sample2D}` providing zero-copy views over `[f64; 2]` slices, `(f64, f64)` slices, paired `(&[f64], &[f64])` slices, and `egui_plot::PlotPoint` slices.
   - Implement stack-allocated `math_commons::fit` (centered and scaled polynomial OLS degrees `1..=4` preserving conditioning at high coordinate offsets, analytical-Jacobian damped Levenberg-Marquardt up to 4 parameters, central finite-difference damped Levenberg-Marquardt up to `MAX_FIT_PARAMS = 6` parameters via `levenberg_marquardt_numeric`, and closed-form seed estimators including stack-sampled Jacquelin sinusoidal seeding) and `math_commons::poi` (`threshold_crossings_into`, `closest_threshold_crossing`, `y_at_x_into`, `parabolic_vertex`, `peak_extremum`, linear-time plateau-aware `detect_local_extrema_into`, fused single-pass `scan_line_pois`, and two-cursor $O(|a|+|b|)$ monotone + AABB general `intersect_polylines_into`).
   - Re-export `fit`, `poi`, and `series` unconditionally from `math_explorer` under default features (`default = []`) with zero GUI dependencies.
2. **Canonical `FitModel` and typed `FitError` diagnostics**:
   - Replace `Polynomial(usize)` with canonical `Linear`, `Quadratic`, `Cubic`, and `Quartic` variants on `FitModel` (`FitType` in `egui_plot`), rejecting degrees outside `1..=4` via `FitError::UnsupportedDegree`.
   - Define `FitError` (`InsufficientPoints`, `UnsupportedDegree`, `InvalidDomain`, `SingularMatrix`, `NonConvergence`) implementing `thiserror::Error` and `diagnostics::Diagnostic` in `math_commons::error` (re-exported in `math_explorer::error`), eliminating silent seed fallbacks on unconverged Levenberg-Marquardt runs.
3. **Delegate UI and simulation tabs to `math_commons` & enforce traceability**:
   - Delete duplicated solver implementations from `egui_plot/src/fit/{lm,nonlinear,poly,seeds}.rs` and `egui_plot/src/poi/interp.rs`, keeping thin UI adapters in `egui_plot` and delegating `harmonic_equilibrium_root`, `compute_epidemic_peak`, and `capacity_fade` directly to `math_explorer::{fit, poi}`.
   - Remove `"crates/math_commons"` from `IGNORED_TRACEABILITY_MEMBERS` in `crates/oxidize_core/src/traceability/runner.rs` so `TraceabilityEngine` enforces `[cite:math_commons]` and `#[verified_engine::verified]` coverage across `math_commons`.

## Consequences

- **Positive**: Headless `math_explorer` exposes verified curve-fitting and critical-point solvers with ~18–20 crates and zero GUI dependencies. Core polynomial, nonlinear LM, crossing, extremum, and intersection solvers execute with zero heap allocations under `VerifiedAllocator::lock_allocations()`.
- **Positive**: `FitModel` satisfies `Eq` and `Hash` without degree-clamping collisions, and `FitError` provides structured diagnostic metadata (`MATH-FIT-001`..`005`).
- **Positive**: Dense stack-matrix elimination and symmetric normal-equation mirroring (`lm`, `poly`, `seeds`) use `split_at_mut` slice iterators rather than range-indexed sub-loops, satisfying strict `clippy::needless_range_loop` invariants with zero heap allocations.
- **Trade-off**: Polynomial regression in `math_commons::fit` is intentionally bounded to stack degrees `1..=4`; higher-degree polynomial systems remain out of scope for zero-allocation stack execution and return `FitError::UnsupportedDegree`.
