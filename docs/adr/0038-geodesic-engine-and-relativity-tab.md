# ADR 0038: Stack-Allocated Geodesic Engine and General Relativity Tab

## Status

Accepted (2026-11-21)

## Context

Previously, [`crates/pure_math/src/pure_math/tensor/christoffel.rs`](../../crates/pure_math/src/pure_math/tensor/christoffel.rs) and [`crates/pure_math/src/pure_math/tensor/metric.rs`](../../crates/pure_math/src/pure_math/tensor/metric.rs) evaluated single-point Christoffel symbols over heap-allocated `nalgebra::DMatrix<f64>` and `DVector<f64>` buffers without ODE trajectory integration, while [`crates/domain_physics/src/physics/high_energy/general_relativity/mod.rs`](../../crates/domain_physics/src/physics/high_energy/general_relativity/mod.rs) exposed only scalar formulas on `SchwarzschildBlackHole`. Neither tensor connection calculus nor curved-spacetime trajectories were surfaced in `math_explorer_gui`, and `VerifiedAllocator` prohibits dynamic heap allocation inside hot simulation loops.

## Decision

1. **Stack-Allocated `StaticMetric<const N: usize>` and `GeodesicSystem` in `pure_math::tensor`:**
   - Added [`static_metric.rs`](../../crates/pure_math/src/pure_math/tensor/static_metric.rs) defining `StaticMetric<const N: usize>`, `StaticRiemannianMetric`, `TwoSphereMetric`, and `static_christoffel_symbols` over fixed-size `SVector<f64, N>` and `SMatrix<f64, N, N>`. Implementations may override `christoffel_at` and `inverse_metric_at` with closed-form formulas while retaining `static_christoffel_symbols` for second-order central-difference cross-validation.
   - Added [`geodesic.rs`](../../crates/pure_math/src/pure_math/tensor/geodesic.rs) defining `GeodesicState`, `GeodesicStepStatus`, and `GeodesicSystem<M, N>` implementing `OdeSystem<ArrayState<4/6/8>>` with zero heap allocations and pre-horizon / polar singularity guards (`r <= 1.02 * R_s`, `sin(theta) <= 1e-4`).
   - Preserved all existing public signatures in `metric.rs`, `christoffel.rs`, and `differentiation.rs`.

2. **Curved Spacetimes and Relativistic Helpers in `domain_physics::physics::high_energy`:**
   - Extended `SchwarzschildBlackHole` (`StaticMetric<4>`), added `KerrEquatorialMetric` (`StaticMetric<4>`), and added `MorrisThorneWormhole` (`StaticMetric<3>`), along with `photon_sphere_radius`, `effective_potential`, `flamm_paraboloid_z`, and `combined_clock_rate` combining `gravitational_time_dilation` with `calculate_lorentz_factor`.

3. **Auto-Discovered `RelativityTab` and Academic Traceability:**
   - Added [`math_explorer_gui/src/tabs/relativity/mod.rs`](../../math_explorer_gui/src/tabs/relativity/mod.rs) (`// @explorer_feature = "physics"`, `#tab=relativity` slug, `"Physics"` academic domain) registering `GeodesicPlaygroundTool`, `ChristoffelInspectorTool`, and four starter gallery presets.
   - Authored [`papers/general_relativity_geodesics.tex`](../../papers/general_relativity_geodesics.tex) with `\usepackage{accessibility}` and linked it in [`traceability.toml`](../../traceability.toml) alongside existing `tensors.tex` and `dwarf_galaxy_empirical_dependencies.tex` mappings.

## Consequences

- 1,000+ `RungeKutta4` geodesic steps execute with zero dynamic heap allocations under `VerifiedAllocator`.
- Closed-form and central-difference Christoffel symbols agree within `1e-6` across Schwarzschild, equatorial Kerr, Morris-Thorne wormhole, and 2-sphere manifolds.
- Users can interactively explore gravitational lensing, ISCO precession, wormhole throat transits, twin clocks, and connection tensors on desktop and `wasm32-unknown-unknown` within the 3 MB WASM bundle budget.
