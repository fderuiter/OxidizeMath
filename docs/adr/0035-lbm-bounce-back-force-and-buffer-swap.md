# ADR 0035: LBM Double-Buffer Swap Consolidation and Zero-Allocation Bounce-Back Force Telemetry

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
The D2Q9 Lattice Boltzmann solver in [`crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/mod.rs) separates pre-streaming distributions (`f`) and post-streaming/post-collision distributions (`f_new`) inside [`LatticeState`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs):

1. **Missing Buffer Swap in Inherent `LatticeBoltzmann::step`**: `SimulationModel::step` invoked `DoubleBufferedState::swap_buffers` after calling `self.step()`, whereas inherent [`LatticeBoltzmann::step`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/algorithms.rs) and `EvolutionEngine::step` omitted `swap_buffers`. Callers invoking inherent `solver.step()` repeatedly streamed from stale `self.state.f` distributions, freezing interior fluid propagation beyond immediate inlet neighbors.
2. **Discarded Bounce-Back Momentum Exchange and Wake Vorticity**: During solid-link bounce-back in `stream()`, reflected populations $f_{\text{opp}(k)}(\mathbf{x}, t)$ were written to `f_new` without accumulating the momentum transfer $\Delta \mathbf{P}_{\text{obs}} = -2 f_{\text{opp}(k)}(\mathbf{x}, t)\,\mathbf{c}_k$ imparted onto solid obstacles, and `LatticeBoltzmann` lacked zero-allocation accessors for net aerodynamic force, central-difference wake curl ($\omega_z = \partial_x u_y - \partial_y u_x$), and active solid cell counts needed by interactive wind-tunnel telemetry.

## Decision
1. **Consolidate `DoubleBufferedState::swap_buffers` Inside `LatticeBoltzmann::step`**:
   - Invoke `DoubleBufferedState::swap_buffers(&mut self.state)` at the end of inherent [`LatticeBoltzmann::step`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/algorithms.rs) immediately after `macroscopic_and_collision()`, and remove the duplicate `swap_buffers` call from `SimulationModel::step`.
   - Inherent `step()`, `SimulationModel::step()`, and `EvolutionEngine::step()` now execute a single pointer-swap per lattice step with zero heap allocations.
2. **Accumulate Zero-Allocation Momentum-Exchange Forces on `LatticeState`**:
   - Add scalar fields `obstacle_force_x: f64` and `obstacle_force_y: f64` to [`LatticeState`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs) (reset on `init_equilibrium` and `clear_obstacles`).
   - Decompose `LatticeBoltzmann::stream` into bounded `#[verified_engine::verified]` interior and boundary helpers (eliminating function-length lint bypasses) that accumulate $-2 f_{\text{opp}(k)}(\mathbf{x}, t)\,(c_{x,k}, c_{y,k})$ across interior and periodic-boundary obstacle bounce-back links while excluding top/bottom channel walls.
3. **Expose Aerodynamic Force, Vorticity Curl, and Solid-Budget Accessors**:
   - Add `#[verified_engine::verified]` methods in [`crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/telemetry.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/telemetry.rs) and [`state.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs): `net_obstacle_force(&self) -> (f64, f64)`, `vorticity_curl(&self, x: usize, y: usize) -> f64`, and `obstacle_cell_count(&self) -> usize`.
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) to record the expanded public solver and state signatures.

## Consequences
- **Positive**: Multi-step `D2Q9` streaming propagates momentum across the entire channel domain identically via inherent `step()`, `SimulationModel`, and `EvolutionEngine`, while preserving strict `VerifiedAllocator` zero-allocation invariants (`NASA Power of 10` Rule 3).
- **Positive**: Downstream consumers such as [`LatticeBoltzmannTool`](../../math_explorer_gui/src/tabs/fluid_dynamics/lattice_boltzmann.rs) obtain live drag/lift coefficients ($C_D, C_L$), signed von Kármán wake vorticity rasters, and solid-area budget counts directly from `domain_physics` without auxiliary grid allocations.
- **Neutral**: Existing callers of `SimulationModel::step` observe identical single-swap semantics because the redundant outer swap was removed in lockstep.
