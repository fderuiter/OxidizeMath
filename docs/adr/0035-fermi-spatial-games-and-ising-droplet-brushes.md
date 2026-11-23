# ADR 0035: Couple `SpinLattice` with Zero-Allocation Fermi Games, Droplet Brushes, and 0D `ReplicatorDynamics` Overlays

## Status
Accepted (extends [ADR 0012](0012-background-simulation-worker-offload.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
Spatial statistical mechanics in [`crates/domain_physics/src/physics/stat_mech/ising.rs`](../../crates/domain_physics/src/physics/stat_mech/ising.rs) and evolutionary game theory in [`crates/domain_applied/src/applied/game_theory/evolutionary/replicator.rs`](../../crates/domain_applied/src/applied/game_theory/evolutionary/replicator.rs) operated in isolation:
1. **Scalar-Only Lattice Evolution**: `SpinLattice::evolve` evaluated only symmetric ferromagnetic/antiferromagnetic scalar couplings `(J, h)`, preventing asymmetric `2x2` strategic interactions (`Prisoner's Dilemma`, `Stag-Hunt`, `Hawk-Dove`) on spatial topologies where cluster boundary reciprocity diverges from well-mixed ODEs.
2. **Non-Interactive GUI Canvas Near File-Length Ceiling**: [`math_explorer_gui/src/tabs/solid_state/ising.rs`](../../math_explorer_gui/src/tabs/solid_state/ising.rs) rendered its `100x100` lattice texture with `egui::Sense::focusable_noninteractive()` inside a `489`-line module and dropped `SimCommand::ApplyBrush` inside [`math_explorer_gui/src/tabs/solid_state/ising/runner.rs`](../../math_explorer_gui/src/tabs/solid_state/ising/runner.rs), blocking direct nucleation experiments via `InteractiveTool::on_brush`.

## Decision
1. **Zero-Allocation Pairwise Fermi Kernel on `SpinLattice`**:
   - Add `SpinLattice::evolve_fermi_game(&mut self, steps: usize, temperature: f64, payoff: &MatrixPayoff)` in [`crates/domain_physics/src/physics/stat_mech/ising.rs`](../../crates/domain_physics/src/physics/stat_mech/ising.rs) coupling `domain_physics` with `domain_applied::applied::game_theory::evolutionary::MatrixPayoff`.
   - Clamp rationality noise `temperature >= 1e-3`, clamp `((pi_i - pi_j) / T)` to `[-500.0, 500.0]` before `f64::exp`, and precompute a stack-allocated `2x5x5` Fermi transition probability table driven by `OxidizeRng` and precomputed von Neumann neighbor indices for zero heap allocations under `VerifiedAllocator`.
2. **Submodule Decomposition, Droplet Brushing, and 0D ODE Comparison**:
   - Extract UI controls, interactive canvas dispatch, and plot overlays into [`math_explorer_gui/src/tabs/solid_state/ising/ui.rs`](../../math_explorer_gui/src/tabs/solid_state/ising/ui.rs) and preset definitions into [`math_explorer_gui/src/tabs/solid_state/ising/preset.rs`](../../math_explorer_gui/src/tabs/solid_state/ising/preset.rs) so every module stays well below the `500`-line ceiling.
   - Switch the lattice `egui::Image` widget to `egui::Sense::click_and_drag()`, route pointer strokes through `InteractiveTool::on_brush`, and handle `SimCommand::ApplyBrush` and `SimCommand::ClearObstacles` in `IsingRunner::process_command` with `[0, 99]` bounds clamping.
   - Retain `Ferromagnetic` mode as default, preserve legacy `IsingState` JSON loading via `#[serde(default)]`, retain `ExportCapabilities::raster_only()` and `"2D Ising Model spin lattice"` accessibility labels, and overlay rolling `100x100` spatial cooperator density against 0D `ReplicatorDynamics` trajectories across `Prisoner's Dilemma`, `Stag-Hunt`, and `Hawk-Dove` presets.

## Consequences
- **Positive**: `SpinLattice::evolve_fermi_game` executes `100x100` spatial evolutionary games with zero post-initialization heap allocations on both native threads and WebAssembly workers while directly contrasting spatial cluster reciprocity against 0D `ReplicatorDynamics` trajectories.
- **Positive**: Interactive circular droplet brushing works both while running and while paused (`SimWorkerState::Ready`), enabling empirical exploration of critical nucleation radii, Stag-Hunt separatrix tipping points, and rationality temperature phase transitions.
- **Neutral**: Reachable public API metadata in `crates/unified_verification/api_baseline.json` is updated via `regenerate-baseline` to record `SpinLattice::evolve_fermi_game`.
