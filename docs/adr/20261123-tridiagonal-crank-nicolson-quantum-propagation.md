# ADR 20261123: Tridiagonal Crank-Nicolson 1D Quantum Propagation and Persistent Unified GIF Stepping

## Status

Accepted

## Context

Prior to this decision, 1D Schrödinger wavepacket evolution in `domain_physics` and the interactive Quantum Wave Simulator (`math_explorer_gui/src/tabs/quantum/wave_sim/`) constructed dense $N \times N$ complex matrices (`QuantumOperator`) and recomputed dense Padé matrix exponentials (`scaled_h.exp()`) whenever `dt` or the spatial potential $V(x)$ changed:

1. **Quadratic Memory & Cubic Reconfiguration Bottleneck**: Representing the nearest-neighbor 3-point finite-difference Laplacian on an $N$-point grid with a dense `DMatrix<Complex<f64>>` consumed $O(N^2)$ memory ($\approx 16\,\text{MB}$ per matrix at $N = 1000$) and required $O(N^3)$ Padé scaling-and-squaring to rebuild $\hat{U}(\Delta t)$ plus $O(N^2)$ matrix-vector multiplications per step, stalling interactive sliders and `SpatialBrushKit` potential sculpting at larger grid resolutions.
2. **Heap Allocations During Parameter Sweeps & Probability Extraction**: Recomputing dense matrix exponentials allocated temporary `DMatrix` buffers on the heap whenever `dt` changed, and `QuantumState::probability_density` allocated a fresh `Vec<f64>` per snapshot rather than writing into preallocated workspace slices.
3. **Dense Real-Symmetric Matrix Exponential Overhead**: General small/medium real-symmetric Hamiltonians in `crates/domain_physics/src/physics/quantum/schrodinger.rs` gated `SymmetricEigen` behind `n >= 4` and otherwise fell back to complex Padé approximation.
4. **Stateless Unified GIF Frame Sampling**: `UnifiedSimTool::step_simulation` instantiated a fresh model `M::new` on every call, preventing `InteractiveTool::sample_animation_frames` from advancing a single continuous trajectory across multi-frame GIF exports and discarding structured `SimulationData::WaveData` payloads when snapshots omitted flat `custom_data` pairs.

## Decision

1. **Tridiagonal Hamiltonian & Zero-Allocation Crank-Nicolson Propagator (`domain_physics`)**:
   - Add `TridiagonalHamiltonian1D` and `CrankNicolsonPropagator1D` in `crates/domain_physics/src/physics/quantum/tridiagonal.rs` and re-export them from `domain_physics::physics::quantum`.
   - Preallocate capacity-locked Thomas sweep buffers (`upper_prime`, `inv_denom`, `rhs`) at construction (`with_capacity`) so both `configure_dt` and `evolve_in_place` execute in $O(N)$ time with zero heap allocations under `VerifiedAllocator` and `VerifiedEngine::run_verified` (NASA Power of 10 Rule 3), including across live `dt` slider adjustments and in-place potential sculpting (`update_from_potential`).
   - Solve the Cayley transform $(I + \frac{i\Delta t}{2\hbar}\hat{H})\psi^{n+1} = (I - \frac{i\Delta t}{2\hbar}\hat{H})\psi^n$, preserving discrete unitarity $\|\psi^{n+1}\|_2 = \|\psi^n\|_2$ to machine precision without pivoting because every forward-sweep pivot satisfies $\operatorname{Re}(\tilde{a}_k) \ge 1$ for real symmetric tridiagonal $\hat{H}$.
2. **Zero-Allocation Born Probability Slice Extraction**:
   - Add `QuantumState::probability_density_in_place(&self, out: &mut [f64])` in `crates/domain_physics/src/physics/quantum/types.rs` so callers populate preallocated buffers without per-frame heap allocations while preserving `QuantumState::probability_density` for convenience callers.
3. **Real-Symmetric Spectral Fast Path in `time_evolution_operator`**:
   - Update `crates/domain_physics/src/physics/quantum/schrodinger.rs` to dispatch any finite real-symmetric $N \times N$ (`N >= 1`) `QuantumOperator` through `nalgebra::SymmetricEigen` ($U = V \operatorname{diag}(e^{-i\lambda_k \Delta t / \hbar}) V^\top$), handling $0 \times 0$ and non-finite inputs safely while retaining `scaled_h.exp()` for general complex Hermitian or non-symmetric operators and keeping all existing `hamiltonian.rs` and `schrodinger.rs` signatures intact.
4. **4-Substep Crank-Nicolson Wave Simulator & 1000-Point Grid Scaling**:
   - Migrate `WaveRunner` (`math_explorer_gui/src/tabs/quantum/wave_sim/runner.rs` and `wave_sim_sculpt.rs`) to `TridiagonalHamiltonian1D` and `CrankNicolsonPropagator1D` preallocated to `MAX_GRID_SIZE` (`1000`), advancing 4 Crank-Nicolson substeps of `dt / 4` per simulation tick to suppress phase dispersion while maintaining zero heap allocations on the stepping hot path.
5. **Persistent Unified GIF Stepping & Structured Snapshot Datasets**:
   - Store a lazy `step_model: Option<M>` on `UnifiedSimTool<M>` (`math_explorer_gui/src/async_sim/unified/tool.rs`) so `step_simulation` reuses a continuous solver instance across `sample_animation_frames` calls and clears `step_model` on resets, state loads, slider edits, and formula updates.
   - Add `UnifiedModel::snapshot_datasets` and `UnifiedSimTool::export_datasets` so unified models backed by `SimulationData::WaveData`, `MultiSeries`, or flat `custom_data` pairs export consistent datasets for CSV/JSON/SVG/PNG/GIF pipelines.

## Consequences

- 1D quantum evolution scales smoothly to `MAX_GRID_SIZE = 1000` in $O(N)$ memory and $O(N)$ operations per substep with zero post-initialization heap allocations, verified under `VerifiedEngine::run_verified` across live `dt` and potential edits.
- Unitary norm conservation holds to machine precision ($\le 10^{-12}$ drift over hundreds of steps), and real-symmetric dense `QuantumOperator` callers gain exact spectral exponentiation without API breakage.
- Multi-frame GIF exports on `UnifiedSimTool` tabs record genuine time evolution rather than repeating the first stepped frame from $t = 0$.
