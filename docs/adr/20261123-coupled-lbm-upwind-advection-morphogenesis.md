# ADR 20261123: Coupled Lattice Boltzmann D2Q9 and First-Order Upwind Advection Morphogenesis Studio

## Status

Accepted

## Context

`OxidizeMath` already provided two independent 2D continuum engines:

1. Isotropic reaction-diffusion morphogenesis (`SchnakenbergKinetics` and `GrayScottKinetics` over `FiniteDifference2D` and `MaskedFiniteDifference2D` with microfluidic `CellMask2D` roles) in `domain_biology`.
2. Two-dimensional `D2Q9` BGK Lattice Boltzmann aerodynamics (`LatticeBoltzmannD2Q9<BgkCollision>`) with Zou-He wind-tunnel inflow/outflow and bounce-back solid obstacles in `domain_physics`.

However, `TuringSystem` evolved concentrations under pure Laplacian diffusion (`D ∇²c + R(u, v)`), preventing simulation of advective-diffusive morphogenesis such as Kármán vortex wakes braiding Gray-Scott autocatalytic spots or viscous laminar transport through `CellMask2D` serpentine microfluidic chicanes.

In addition, `domain_biology` and `domain_physics` are independent domain crates in the workspace dependency graph, `LatticeBoltzmannState` stores fields in column-major layout (`idx(x, y) = x * height + y`) whereas `FiniteDifference2D` and `CellMask2D` use row-major indexing (`flatten_2d_index(x, y, width) = y * width + x`), and `#![no_std]` `engine_pde::StencilOperators` only pre-cached `inv_dx`, `inv_dx_sq`, `inv_dy_sq`, and `inv_dz_sq` (omitting `inv_dy`).

## Decision

1. **`#![no_std]` Stencil Reciprocal & Pure-Math 2D Donor-Cell Upwind Advection** (`engine_pde`, `pure_math`):
   - Added `inv_dy: f64` (`1.0 / *dy`) to `StencilOperators` in [stencil.rs](../../crates/02_engines/engine_pde/src/stencil.rs) (re-exported via [pure_math_facade.rs](../../crates/04_bindings/oxidizemath/src/pure_math_facade.rs)) and reused `self.inv_dy` across `upwind_2d_at` and `upwind_3d_at`.
   - Created [advection_2d.rs](../../crates/02_engines/engine_pde/src/advection_2d.rs) providing conservative face-velocity donor-cell upwind divergence (`-∇ · (v c)`), `advection_courant_number`, `advection_diffusion_cfl_limit`, and `grid_peclet_number`, all verified zero-allocation under `#[verified_engine::verified]`.

2. **Wall-Aware Patankar Advection-Diffusion-Reaction Stepping** (`domain_biology`):
   - Created [advection.rs](../../crates/03_domains/domain_biology/src/biology/diffusion/fd2d/advection.rs) defining `AdvectedSpatialDiffusion<N>` for `FiniteDifference2D` and `MaskedFiniteDifference2D`.
   - On unmasked grids (`!mask.has_active_cells()`), explicit conservative donor-cell fluxes preserve discrete mass to machine precision (`< 1e-12`) under zero-normal-flux outer boundaries.
   - On sculpted grids (`mask.has_active_cells()`), solid `ZeroFluxWall` faces enforce zero normal diffusive and advective flux (`u_face = 0`, `c_neighbor = c_center`) while outflow flux sinks are folded into the semi-implicit Patankar denominator so morphogen concentrations remain strictly non-negative (`c >= 0.0`) without wall leakage.
   - Created [advection.rs](../../crates/03_domains/domain_biology/src/biology/morphogenesis/advection.rs) exposing `TuringSystem::step_advected(dt, ux, uy)` over borrowed row-major velocity slices, preserving strict crate independence between `domain_biology` and `domain_physics`.

3. **Zero-Allocation Column-to-Row-Major LBM Projection** (`domain_physics`):
   - Extended [aerodynamics.rs](../../crates/03_domains/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/aerodynamics.rs) with `project_velocity_row_major_into`, `project_obstacles_row_major_into`, and `sync_obstacles_row_major`, translating internal `D2Q9` column-major state into row-major buffers via `flatten_2d_index` with zero heap allocations.

4. **GUI Multiphysics Co-Stepping, Serpentine Portal Sealing, and CFL Guard** (`math_explorer_gui`):
   - Created [advection.rs](../../apps/math_explorer_gui/src/tabs/morphogenesis/advection.rs) hosting `MorphoWindTunnel` (pre-allocated `LatticeBoltzmannD2Q9<BgkCollision>`, `ux_buf`, `uy_buf`, and `obstacle_buf`) inside `MorphogenesisKineticsRunner`.
   - Synchronized `ReactionCellRole::ZeroFluxWall` edits (pointer `BrushStroke`, keyboard reticle `Arrow` + `Space`/`Enter`, `stamp_microfluidic_maze`, and RLE preset loads) directly into `D2Q9` bounce-back obstacles, sealing dead-end side portals when alternating serpentine baffles are stamped so channel flow traverses all three passes.
   - Added wind-tunnel toggle, `inlet_u` and `viscosity` sliders, live Péclet (`Pe`), Courant (`Co`), and combined advective-diffusive CFL readouts (using `SemanticPalette`), keyboard-accessible `"⚡ Stabilize Step (CFL)"` recovery with `TwinSlotLiveRegion` announcements, and the `"Vortex Wake Morphogenesis"` starter gallery preset while keeping default advection fields omitted (`#[serde(skip_serializing_if = ...)]`) for backward-compatible `< 4 KiB` URL share payloads.

```text
LatticeBoltzmannD2Q9 (col-major) ──project_velocity_row_major_into──▶ ux_buf, uy_buf (row-major)
        ▲                                                                      │
sync_from_cell_mask                                                            ▼
        │                                                       TuringSystem::step_advected
CellMask2D (ZeroFluxWall) ───────────────────────────────▶ MaskedFiniteDifference2D (Patankar)
```

## Consequences

- **Domain Modularity**: `domain_biology` and `domain_physics` remain completely decoupled in `Cargo.toml`; multiphysics composition occurs strictly over flat `&[f64]` velocity and `&[bool]` obstacle slices in `math_explorer_gui`.
- **Zero-Allocation Hot Loop**: Both isolated `domain_biology` advected stepping and 50-step coupled `MorphogenesisKineticsRunner` fluid-reaction-diffusion execution run with zero heap allocations under `VerifiedAllocator`.
- **Scope Boundaries**: Three-dimensional Navier-Stokes coupling and reactive buoyancy feedback from morphogen concentration onto fluid momentum remain out of scope for this iteration; time-travel `DualField2D` checkpointing yields (`None`) while live LBM populations are active and resumes `K = 16` stride when wind-tunnel advection is disabled.

## Verification

- `cargo test -p engine_pde`
- `cargo test -p pure_math advection_2d`
- `cargo test -p domain_physics lattice_boltzmann`
- `cargo test -p domain_biology --test test_advected_morphogenesis_2d`
- `cargo test -p math_explorer_gui --bins advection_tests -- --test-threads=1`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
