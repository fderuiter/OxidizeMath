# ADR 0035: Zero-Allocation 2D Chladni Plate and Wave Tank Solver

## Status

Accepted

## Context

OxidizeMath previously modeled 1D partial differential equations (`pure_math::analysis::pde`), closed-form Fraunhofer slit intensities (`domain_physics::optics::diffraction`), and 2D Navier-Stokes fluid grids, without a unified interactive 2D wave-propagation and thin-plate modal laboratory. Simulating both second-order hyperbolic acoustic/optical waves and fourth-order Kirchhoff-Love biharmonic Chladni plates interactively at 60 FPS on both native desktop and `wasm32-unknown-unknown` requires:

1. **Zero dynamic heap allocations** inside time-stepping, Chladni sand particle drift, Marching-Squares nodal contour extraction, and 2D FFT spectral decomposition under `VerifiedAllocator::lock_allocations()` (NASA Power of 10 Rule 3).
2. **Unconditional explicit substep stability** across both second-order Courant-Friedrichs-Lewy bounds ($c \Delta t / \min(\Delta x, \Delta y) \le 0.65$) and fourth-order biharmonic Fourier stability bounds ($\sqrt{D} \Delta t (\Delta x^{-2} + \Delta y^{-2}) \le 0.18$).
3. **Zero new external Cargo dependencies** and adherence to the `< 3 MiB` compressed WASM bundle budget.

## Decision

1. **`Wave2DSolver` in `crates/pure_math/src/pure_math/analysis/pde/wave_2d.rs`**:
   - Reuses `StencilOperators::new_2d` (`fused_stepper.rs`) and `flatten_2d_index` over pre-allocated `Vec<f64>` displacement (`u_prev`, `u_curr`, `u_next`), intermediate Laplacian (`laplacian_buf`), energy envelope (`energy_ema`), and phase-demodulated modal (`modal_field`) buffers.
   - Evaluates fourth-order Kirchhoff-Love plate bending ($\nabla^4 u = \nabla^2(\nabla^2 u)$) in two cache-contiguous 5-point stencil passes without allocating temporary grids.
   - Supports `Dirichlet`, `Neumann`, `Clamped`, and cubic `Sponge` perimeter boundaries alongside spatial `wall_mask` (nodal clamps and multi-slit barriers) and `wave_speed_mask` ($s(x, y) = 1 / n(x, y)$ refractive lenses).
2. **`SpectralAnalyzer2D` (`wave_2d/spectral.rs`)**:
   - Plans row and column forward 1D FFTs once at construction via `rustfft::FftPlanner` and executes `process_with_scratch` against pre-allocated complex spatial, transpose, and scratch buffers.
3. **`ChladniSandSystem` and `MarchingSquares2D` (`wave_2d/particles_and_contours.rs`)**:
   - Seeds `4,000` SoA tracer grains deterministically via `oxidize_core::rng::OxidizeRng` and drifts grains down $-\nabla \langle u^2 \rangle$ with amplitude-proportional stochastic agitation onto nodal Chladni figures.
   - Extracts zero-level nodal isolines into a fixed-capacity segment buffer.
4. **`ChladniWaveTankTool` in `math_explorer_gui/src/tabs/analysis/chladni_wave_tank.rs`**:
   - Registers `ChladniWaveTankTool` and four `StarterPresetMetadata` cards (**Square Chladni Plate**, **Circular Drumhead**, **Double-Slit Interference**, and **Acoustic Lens**) in `AnalysisTab` with URL hash persistence, `ToolStateBounds` JSON validation, `AccessKit` keyboard navigation, and `SVG`, `PNG`, `CSV`, and `JSON` exports.

## Consequences

- **Positive**: Verified eigenfrequencies of clamped square Kirchhoff-Love plates match Leissa (1969) analytic eigenvalues within `2%`, double-slit detector cross-sections reproduce analytic Young first-order fringe spacing within `1.0` cell on `128x128` grids, and 100 consecutive wave, sand, and 2D FFT steps execute with zero dynamic allocations.
- **Trade-offs**: Explicit Verlet integration of fourth-order biharmonic plates requires $\Delta t_{\text{sub}} = O(h^2 / \sqrt{D})$, bounded deterministically by `safe_substep_plan` (`MAX_VERLET_SUBSTEPS = 64`).
