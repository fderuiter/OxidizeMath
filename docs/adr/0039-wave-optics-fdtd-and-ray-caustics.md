# ADR 0039: Pre-Allocated 2D FDTD Wave Solver, Hamiltonian Ray Caustics, and Auto-Discovered Optics Playground

## Status
Accepted (extends [ADR 0008](0008-finite-state-solver-guard.md), [ADR 0012](0012-background-simulation-worker-offload.md), [ADR 0032](0032-high-contrast-focus-selection-and-raster-tokens.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
Prior to this decision, `pure_math::analysis::pde::wave` modeled only 1D scalar wave formulas (`WaveEquation1D` and `StringWaveSolution`), `domain_physics::physics::optics` contained only two scalar radiometric/range-resolution formulas in [`sensing.rs`](../../crates/domain_physics/src/physics/optics/sensing.rs) with a placeholder `DUMMY = 1.0` verification block in [`optics/mod.rs`](../../crates/domain_physics/src/physics/optics/mod.rs) and a mismatched `graph_parameters_rust.tex` binding in [`traceability.toml`](../../traceability.toml), and `math_explorer_gui` exposed zero interactive 2D wave or optical physics tools.

Bridging physical wave diffraction and geometric-optics caustics across desktop and `wasm32` targets introduces three engineering constraints:
1. **Zero-Allocation Hot Loops Under `VerifiedAllocator`**: Continuous 2D FDTD stepping and multi-ray Runge-Kutta integration must allocate all field, sponge, metamaterial, trajectory, and caustic buffers up front so simulation frames incur zero dynamic heap allocations.
2. **Negative-Index Metamaterial Fidelity ($n < 0$)**: A non-dispersive scalar equation proportional to $n(x, y)^2$ cannot distinguish a Veselago negative-index slab ($n = -1$) from vacuum ($n = +1$). Time-domain FDTD requires causal dispersive auxiliary fields, while geometric ray tracing requires generalized Snell momentum updates across sign-inverting interfaces.
3. **Accessible Multi-Theme Visualization and Calibrated Sensing**: Wavefront rasters, geometric ray bundles, caustic cusp markers, and movable virtual detector planes must maintain WCAG 2.2 AA (`>= 3.0:1`) contrast across `Dark`, `Light`, and `HighContrast` themes, expose `Role::Image` alternative text and trap-free keyboard navigation, and pair diffractive irradiance profiles with calibrated `lambertian_intensity` and `lws_range_resolution` metrics.

## Decision
1. **Zero-Allocation 2D FDTD Solver (`WaveField2D`) in `pure_math`**:
   - Add [`wave_2d.rs`](../../crates/pure_math/src/pure_math/analysis/pde/wave_2d.rs) and re-export `WaveField2D`, `WaveFieldConfig2D`, `WaveSource2D`, and `WaveSourceKind` from [`wave.rs`](../../crates/pure_math/src/pure_math/analysis/pde/wave.rs).
   - Step positive-index inhomogeneous media via centered leapfrog using [`StencilOperators::central_diff_2nd_2d`](../../crates/pure_math/src/pure_math/analysis/pde/fused_stepper.rs), quadratic sponge absorbing boundaries, Dirichlet obstacle masks, and 2D Courant-Friedrichs-Lewy (`cfl_dt`) clamping.
   - Pre-allocate passive matched-Drude auxiliary fields (`hx`, `hy`, `jz`, `mx`, `my`, `omega_p_sq`) activated automatically when negative-index cells ($n < 0$) are configured, reproducing backward phase fronts and Veselago superlens focusing with zero heap allocation during `step`.
2. **Optical Media, RK4 Hamiltonian Caustics, and Virtual Detector Readouts in `domain_physics`**:
   - Add [`refraction.rs`](../../crates/domain_physics/src/physics/optics/refraction.rs) providing `OpticsPreset` (`DoubleSlit`, `LuneburgLens`, `MaxwellFishEye`, `BiconvexDroplet`, and `VeselagoLens`), `OpticalMediumConfig`, `snell_refracted_angle`, and `snell_refract_vector`.
   - Add [`caustics.rs`](../../crates/domain_physics/src/physics/optics/caustics.rs) implementing `OdeSystem<ArrayState<4>>` for `OpticalMediumConfig` and `HamiltonianRayTracer` backed by [`RungeKutta4`](../../crates/pure_math/src/pure_math/analysis/ode/solvers.rs), stack-resident interface bisection, and ray-tube Jacobian sign-flip caustic detection.
   - Extend [`sensing.rs`](../../crates/domain_physics/src/physics/optics/sensing.rs) with `DetectorConfig`, `DetectorReadout`, and `sample_detector_line_in_place` coupling 2D FDTD irradiance profiles to `lambertian_intensity` and `lws_range_resolution`.
   - Bind `optics = "spec:wave_optics_caustics_v1"` in [`traceability.toml`](../../traceability.toml) and verify physical Snell, Luneburg, Veselago, and Lambertian invariants in [`optics/mod.rs`](../../crates/domain_physics/src/physics/optics/mod.rs).
3. **Auto-Discovered `OpticsWaveTab` Playground in `math_explorer_gui`**:
   - Add [`math_explorer_gui/src/tabs/optics_waves/`](../../math_explorer_gui/src/tabs/optics_waves/mod.rs) annotated with `// @explorer_feature = "physics"` for `build.rs` registration on `SimulationFramework`, bounded to `64..=256` grids, backed by `BackgroundWorker<OpticsWaveRunner>`, `FiniteStateGuard` recovery, `StarterPresetMetadata` cards, `#tab=optics_waves` lossless URL sharing, contrast-verified colormaps/overlays, and `Role::Image` keyboard-focusable detector controls.

## Consequences
- **Positive**: Users and downstream `math_explorer` consumers gain verified 2D scalar/metamaterial FDTD wave propagation, RK4 Hamiltonian ray caustics, and calibrated optical detector readouts on both native desktop and `wasm32` with zero hot-loop heap allocations.
- **Positive**: `OpticsWaveTab` integrates seamlessly into `SimulationFramework` preset, raster export, URL state, and accessibility contract suites without manual edits to `app.rs`.
- **Neutral**: Adding `OpticsWaveTab`, `OpticsWaveTool`, and three starter presets increments workspace GUI inventory contract counts to `22` tabs, `63` tools, and `77` starter presets, reflected in `api_baseline.json` and contract tests.
