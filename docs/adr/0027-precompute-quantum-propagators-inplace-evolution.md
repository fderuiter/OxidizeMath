# ADR 0027: Precompute Quantum Propagators for Zero-Allocation Stepping

## Status
Accepted

## Context
In `crates/domain_physics/src/physics/quantum/schrodinger.rs`, `evolve_state` invoked `time_evolution_operator` on every simulation step, recomputing the matrix exponential $\hat{U}(\Delta t) = \exp(-i\hat{H}\Delta t/\hbar)$ via `scaled_h.exp()` and allocating a new `QuantumState`. Similarly, `QuantumState::normalize` in `crates/domain_physics/src/physics/quantum/types.rs` allocated a new `DVector<Complex<f64>>` on every call.

In the GUI simulation loops:
1. `WaveUnified::step` (`math_explorer_gui/src/tabs/quantum/wave_sim.rs`) evolved a 200-point 1D grid under a time-independent Hamiltonian by calling `evolve_state` and `normalize` on every tick. At $N = 200$, recomputing the $200 \times 200$ complex matrix exponential cost hundreds of milliseconds and 70 heap allocations (~23.25 MiB) per step, stalling 5-step worker frames and violating NASA Power of 10 Rule 3 (no post-initialization dynamic heap allocations in `VerifiedAllocator`).
2. `SpinVisualizer::step` (`math_explorer_gui/src/tabs/quantum/spin_viz.rs`) reconstructed the $2 \times 2$ Zeeman Hamiltonian and called `evolve_state` and `normalize` on every unpaused UI frame even when `b_field` and `dt` were unchanged.

## Decision
1. **In-Place Unitary State Evolution (`evolve_state_in_place`)**: Added `#[verified_engine::verified] pub fn evolve_state_in_place(state: &mut QuantumState, propagator: &QuantumOperator, scratch: &mut DVector<Complex<f64>>)` in `crates/domain_physics/src/physics/quantum/schrodinger.rs` (and re-exported it in `crates/domain_physics/src/physics/quantum/mod.rs`). It asserts square operator dimensions matching `state.vector.len()` and `scratch.len()`, multiplies `propagator.matrix.mul_to(&state.vector, scratch)`, and swaps `state.vector` with `scratch` without heap allocations.
2. **In-Place State Normalization (`QuantumState::normalize_in_place`)**: Added `#[verified_engine::verified] pub fn normalize_in_place(&mut self)` in `crates/domain_physics/src/physics/quantum/types.rs`, leaving zero-norm vectors unchanged and dividing elements in place by `Complex::new(norm, 0.0)` to preserve exact numerical and non-finite propagation semantics with `QuantumState::normalize`.
3. **Cached Propagator and Scratch Buffer in `WaveUnified`**: Extended `WaveUnified` with `propagator`, `cached_dt`, and `scratch`, initializing them in `init_system` (`cached_dt = None` on initialization and `SimCommand::Reset`) and refreshing `propagator` via `time_evolution_operator(&self.hamiltonian, dt, 1.0)` only when `cached_dt != Some(dt)` after validating `dt.is_finite() && dt > 0.0`. Updated `WaveUnified::step` to call `evolve_state_in_place` and `normalize_in_place`.
4. **Cached Propagator and Scratch Buffer in `SpinVisualizer`**: Extended `SpinVisualizer` with `propagator`, `cached_key: Option<[u64; 4]>`, and a 2-element `scratch` vector, rebuilding the Zeeman Hamiltonian and `propagator` only when the `[bx, by, bz, dt]` IEEE-754 bit patterns change, and advancing `self.psi` in place.
5. **Baseline & Verification Updates**: Regenerated `crates/unified_verification/api_baseline.json` and added `VerifiedAllocator` / `VerifiedEngine::run_verified` zero-allocation and numerical parity tests in `crates/domain_physics/tests/test_quantum.rs`, along with cache invalidation and recovery tests in `wave_sim.rs`, `spin_viz.rs`, and `recovery_tests.rs`.

## Consequences
- **Positive**: Steady-state quantum evolution steps perform a single $O(N^2)$ matrix-vector product and $O(N)$ normalization pass with zero heap allocations, passing `VerifiedEngine::run_verified` under `VerifiedAllocator` (NASA Power of 10 Rule 3).
- **Positive**: `WaveUnified::step` avoids repeated $200 \times 200$ matrix exponentiations when `dt` and `potential_type` are constant, while immediately refreshing the propagator when `dt` changes or `SimCommand::Reset` rebuilds the Hamiltonian.
- **Neutral**: `evolve_state`, `time_evolution_operator`, and `QuantumState::normalize` retain their exact signatures and behavior for existing callers and doctests, and `FiniteStateGuard` divergence recovery for invalid `dt`, `psi`, and `psi_norm` remains unchanged.
