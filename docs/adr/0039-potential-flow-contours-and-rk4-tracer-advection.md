# ADR 0039: Potential Flow Iso-$\psi$ Contours, Stagnation Detection, and `RungeKutta4` Dye Tracer Coupling

## Status
Accepted (extends [ADR 0016](0016-ode-multi-solver-comparative-overlays.md), [ADR 0032](0032-high-contrast-focus-selection-and-raster-tokens.md), and [ADR 0035](0035-iterative-ast-glsl-transpiler-and-shader-exports.md))

## Context
[`PotentialFlowField`](../../crates/domain_physics/src/physics/fluid_dynamics/potential_flow.rs) implemented analytic `velocity`, `stream_function`, and `potential` methods across `UniformFlow`, `Source`, `Vortex`, and `Doublet`, while [`FluidParticleSystem`](../../crates/domain_physics/src/physics/fluid_dynamics/solver.rs) implemented `OdeSystem<FlowState>` and `TimeStepper<FlowState>` over [`Euler`](../../crates/domain_physics/src/physics/fluid_dynamics/conservation.rs). However, `math_explorer_gui` rendered only coarse velocity `Arrows` and required sidebar spinbox coordinate entry, leaving `stream_function` and `FluidParticleSystem` uncoupled from interactive potential-flow exploration. Naive grid contouring across `Source` singularities also produces spurious contour bundles across `atan2` branch cuts and inside `TOLERANCE_HIGH` singularity cores.

## Decision
1. **Singularity-Aware Iso-$\psi$ Contouring and Stagnation Detection**:
   - Add `FlowElement::singularity_center`, `PotentialFlowField::is_near_singularity`, `PotentialFlowField::extract_streamline_segments`, and `PotentialFlowField::detect_stagnation_points` in [`potential_flow.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/potential_flow.rs) backed by `#[verified_engine::verified]` marching-squares and damped Newton-Raphson kernels in [`analysis.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/analysis.rs).
   - Reject marching-squares cells inside `TOLERANCE_HIGH` / cell-core radii and cells where `stream_function` jumps violate Cauchy-Riemann edge continuity across `atan2` branch cuts.
2. **Lagrangian `RungeKutta4` Dye Advection Coupled with `FluidParticleSystem` and `Euler`**:
   - Introduce `PotentialTracerSystem` and `TracerProbeReadout` in [`solver.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/solver.rs), implementing `OdeSystem<Vector2<f64>>` and `TimeStepper<Vector2<f64>>` with `RungeKutta4` and clamping speeds near singularity cores.
   - Couple `PotentialTracerSystem::evaluate_probe` with `FluidParticleSystem<'_, Euler>` and Bernoulli's relation to evaluate convective acceleration $(\mathbf{u} \cdot \nabla)\mathbf{u}$, static pressure, and pressure coefficient $C_p$.
3. **Modularized Tactile Wind-Tunnel GUI and WCAG 2.2 AA Reticle Navigation**:
   - Split `math_explorer_gui/src/tabs/fluid_dynamics/potential_flow` into `{mod,cache,canvas,controls,presets,shader,tests,wasm_tests}.rs` (`<= 500` lines per file), adding canvas handle dragging and stamping, dynamic `Plot::allow_drag` pan locking, keyboard reticle nudging and live-region announcements, GLSL shader export integration, and Kutta-Joukowski lifting cylinder, Rankine oval, and Kelvin-Stuart vortex street presets in [`starter_gallery/fluid_presets.rs`](../../math_explorer_gui/src/starter_gallery/fluid_presets.rs) re-exported by [`starter_gallery.rs`](../../math_explorer_gui/src/starter_gallery.rs).

## Consequences
- **Positive**: `PotentialFlowField::stream_function`, `FluidParticleSystem`, and `conservation::Euler` power live iso-$\psi$ contours, stagnation markers, animated `RungeKutta4` dye streaks, and Bernoulli/convective probe telemetry without `atan2` branch-cut artifacts.
- **Positive**: Every source file remains within the 500-line ceiling and every verified kernel remains within the 60-statement / 60-line limit, synchronized in [`api_baseline.json`](../../crates/unified_verification/api_baseline.json).
- **Neutral**: Existing `PotentialFlowState` JSON presets remain backward-compatible via `#[serde(default)]` fields while preserving `#[serde(deny_unknown_fields)]` validation.
