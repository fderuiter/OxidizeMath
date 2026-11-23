# 40. Couple Lattice Boltzmann and Potential Flow to 2D Upwind Morphogenesis

Date: 2026-11-22

## Status

Accepted

## Context

Previously, 2D morphogenesis inside [`math_explorer_gui/src/tabs/morphogenesis`](../../math_explorer_gui/src/tabs/morphogenesis) relied exclusively on [`FiniteDifference2D`](../../crates/domain_biology/src/biology/diffusion/fd2d/mod.rs) with zero advective velocity, ignored `SimCommand::ApplyBrush` and `SimCommand::ClearObstacles`, and supported only `SchnakenbergKinetics` in production (`GrayScottKinetics` existed only inside an integration test). In parallel, inherent `LatticeBoltzmann::step` in [`crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/algorithms.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/algorithms.rs) omitted `DoubleBufferedState::swap_buffers`, preventing downstream streaming when invoked outside `SimulationModel::step`, and `StencilOperators` in [`crates/pure_math/src/pure_math/analysis/pde/fused_stepper.rs`](../../crates/pure_math/src/pure_math/analysis/pde/fused_stepper.rs) lacked vertical 1st-order reciprocals (`inv_dy`, `inv_two_dy`), 2D upwind advection operators, and solid obstacle neighbor reflection.

Coupling fluid wakes and inviscid vortex flows to biological pattern formation requires conservative, zero-flux 2D upwind transport that strictly obeys NASA Power of 10 zero-allocation hot-loop rules (`VerifiedAllocator`), isolated domain feature compilation (`--no-default-features --features biology` vs `physics`), backward-compatible preset serialization, and WCAG 2.2 `>= 3.0:1` HighContrast graphical contrast.

## Decision

1. **Inherent LBM Buffer Swap & Zero-Allocation Row-Major Transpose (`domain_physics`)**:
   - Move `DoubleBufferedState::swap_buffers(&mut self.state)` into inherent `LatticeBoltzmann::step` and remove the duplicate swap from `SimulationModel::step` so all callers (`LatticeBoltzmann::step`, `SimulationModel::step`, `EvolutionEngine::step`, and `LbmUnified::step`) stream fluid across the entire lattice.
   - Add `LatticeState::copy_macroscopic_row_major` in [`crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs) to transpose column-major `nalgebra::DMatrix(height, width)` macroscopic velocity and obstacle fields into row-major `(y * width + x)` slices with zero heap allocations.

2. **Conservative 2D Upwind & Obstacle-Reflecting Stencils (`pure_math` & `domain_biology`)**:
   - Precompute `inv_dy` and `inv_two_dy` inside `StencilOperators` and add `central_diff_1st_y`, `upwind_flux_y`, `upwind_flux_2d`, `upwind_divergence_2d`, `reflect_2d_neighbors`, `advection_face_velocities`, and `FusedStencilStepper::step_2d_coupled_advection_neumann`.
   - Implement `AdvectionDiffusion2D` in [`crates/domain_biology/src/biology/diffusion/advection_2d.rs`](../../crates/domain_biology/src/biology/diffusion/advection_2d.rs) over preallocated row-major velocity and obstacle buffers, enforcing zero normal diffusive and advective flux across domain edges and internal solid obstacles, discrete vorticity $\omega_z = \partial_x u_y - \partial_y u_x$, and bounded CFL sub-stepping.
   - Promote `GrayScottKinetics` with `#[derive(Theory)]` metadata into [`crates/domain_biology/src/biology/morphogenesis/reaction.rs`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs).

3. **Feature-Gated Fluid Coupling, Sculpting & Accessible Dual Overlays (`math_explorer_gui` & `math_commons`)**:
   - Couple `LatticeBoltzmannD2Q9` and `PotentialFlowField` behind `#[cfg(feature = "physics")]` inside `MorphogenesisRunner`, handling `SimCommand::ApplyBrush` (obstacle and reagent sculpting) and `SimCommand::ClearObstacles` while keeping `step()` allocation-free.
   - Add diverging vorticity (`vorticity_color`, `vorticity_color_light`, `vorticity_color_high_contrast`) and composite (`dual_concentration_vorticity_color`) helpers in [`crates/math_commons/src/math_kernel/colormap.rs`](../../crates/math_commons/src/math_kernel/colormap.rs), live Péclet ($\mathrm{Pe}$) and Damköhler ($\mathrm{Da}$) readouts, and backward-compatible `#[serde(default)]` preset fields alongside two new advected starter presets (`karman_turing_wake` and `gray_scott_spiral_flow`).

## Consequences

- **Positive**: Users can sculpt bluff obstacles and chemical nuclei in real time, watch Kármán vortex wakes and potential-flow circulation shear Schnakenberg spots and Gray-Scott filaments, inspect live $\mathrm{Pe}$ and $\mathrm{Da}$ dimensionless regimes, and toggle concentration / vorticity $\omega_z$ / dual overlays across `Dark`, `Light`, and `HighContrast` themes.
- **Positive**: `LbmUnified` and `LatticeBoltzmann::step` now transport fluid downstream consistently across all solver entry points without post-initialization heap allocations.
- **Trade-offs**: First-order donor-cell upwinding introduces numerical diffusivity $O(|u|\Delta x / 2)$ at high cell Péclet numbers, bounded by adaptive CFL sub-stepping (`MAX_CFL_SUBSTEPS = 64`) to guarantee stability and positivity.
