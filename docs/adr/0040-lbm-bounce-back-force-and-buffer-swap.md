# ADR 0040: LBM Double-Buffer Swap Consolidation and Zero-Allocation Bounce-Back Force Telemetry

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md), [ADR 0037](0037-retire-physics-compatibility-shims.md), and [ADR 0038](0038-federated-inverse-challenges-and-hud.md))

## Context
The D2Q9 Lattice Boltzmann solver in [`crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/mod.rs) separates pre-streaming distributions (`f`) and post-streaming/post-collision distributions (`f_new`) inside [`LatticeState`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs):

1. **Single-Swap Contract Between Inherent `step` and `DoubleBufferedEvolutionEngine`**: Inherent [`LatticeBoltzmann::step`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/algorithms.rs) must swap `f` and `f_new` once per step via `DoubleBufferedState::swap_buffers`, while `EvolutionEngine::step` must delegate to `step_unswapped()` so `DoubleBufferedEvolutionEngine::step_buffered` performs the outer swap without double-swapping.
2. **Unified Boundary Bounce-Back Momentum Exchange and Wake Vorticity**: Interior and periodic-boundary obstacle bounce-back reflections transfer momentum $\Delta \mathbf{P}_{\text{obs}} = -2 f_{\text{opp}(k)}(\mathbf{x}, t)\,\mathbf{c}_k$ onto solid cells, while interactive wind-tunnel visualization and drag challenges require zero-allocation accessors for instantaneous force, central-difference wake curl ($\omega_z = \partial_x u_y - \partial_y u_x$), and solid obstacle cell counts.

## Decision
1. **Unify `step()` and `step_unswapped()` Buffer Swapping**:
   - Execute `stream()`, `boundary_conditions()`, and `macroscopic_and_collision()` inside `step_unswapped(&mut self)` (used by `EvolutionEngine::step`), and invoke `DoubleBufferedState::swap_buffers(&mut self.state)` once at the end of inherent `LatticeBoltzmann::step(&mut self)`.
2. **Accumulate Interior and Periodic-Boundary Bounce-Back Forces on `LatticeState`**:
   - Decompose `LatticeBoltzmann::stream` into bounded `#[verified_engine::verified]` `stream_interior` and `stream_boundary_cell` helpers (eliminating function-length lint bypasses) that accumulate $-2 f_{\text{opp}(k)}(\mathbf{x}, t)\,(c_{x,k}, c_{y,k})$ across both interior and periodic-boundary obstacle links (excluding top/bottom channel walls) into `last_drag_force`, `last_lift_force`, and `has_stepped_forces` (reset on `init_equilibrium` and `clear_obstacles`).
3. **Expose Unified Aerodynamic Force, Vorticity Curl, and Solid-Budget Accessors**:
   - Provide `#[verified_engine::verified]` methods in [`telemetry.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/telemetry.rs), [`forces.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/forces.rs), and [`state.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/lattice_boltzmann/state.rs): `net_obstacle_force(&self) -> (f64, f64)`, `hydrodynamic_forces(&self) -> (f64, f64)`, `vorticity_curl(&self, x: usize, y: usize) -> f64`, and `obstacle_cell_count(&self) -> usize`.
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) to record the public solver and state signatures.

## Consequences
- **Positive**: Multi-step `D2Q9` streaming propagates momentum across the entire channel domain identically via inherent `step()` and `DoubleBufferedEvolutionEngine::step_buffered`, while preserving strict `VerifiedAllocator` zero-allocation invariants (`NASA Power of 10` Rule 3).
- **Positive**: Downstream consumers ([`LatticeBoltzmannTool`](../../math_explorer_gui/src/tabs/fluid_dynamics/lattice_boltzmann.rs) and [`LbmFairingEvaluator`](../../math_explorer_gui/src/tabs/fluid_dynamics/lbm_fairing.rs)) share a single zero-allocation force state for live $C_D, C_L$ telemetry, signed von Kármán wake vorticity rasters, and inverse fairing challenges.
