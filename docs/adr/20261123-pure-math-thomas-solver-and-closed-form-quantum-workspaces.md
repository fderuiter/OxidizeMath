# ADR 20261123: `core`-Slice Thomas Solver, `CrankNicolson1D` Workspace, and Closed-Form $2\times 2$ Quantum Helpers

## Status

Accepted

## Context

While [ADR 20261123: Tridiagonal Crank-Nicolson 1D Quantum Propagation](20261123-tridiagonal-crank-nicolson-quantum-propagation.md) introduced tridiagonal 1D quantum propagation in `domain_physics`, several architectural gaps remained across `pure_math`, `domain_physics`, and `math_explorer_gui`:

1. **Missing Reusable `no_std`-Compatible Thomas Solver in `pure_math`**: Complex tridiagonal LU factorization and forward/backward Thomas sweeps lived exclusively inside `domain_physics::physics::quantum::tridiagonal` instead of serving as reusable `core`-slice PDE primitives alongside `FusedStencilStepper` in `pure_math::pure_math::analysis::pde`.
2. **Post-Initialization Heap Allocations in Two-State & Wavepacket Helpers**: `HydrogenicSuperposition3D::rebuild_propagator` (and `reconfigure_basis`) allocated heap `DMatrix` buffers via `construct_two_state_eigen_hamiltonian` and `time_evolution_operator`, violating NASA Power of 10 Rule 3 (`VerifiedAllocator`) whenever `dt`, dipole coupling, or orbital quantum numbers changed after initialization. Likewise, `gaussian_wavepacket` allocated a fresh `DVector<Complex<f64>>` on wavepacket relaunches in `WaveRunner`.
3. **Dense Matrix Exponential Overhead on $2 \times 2$ Hermitian Systems**: `time_evolution_operator` ran general `SymmetricEigen` or complex Padé scaling-and-squaring (`scaled_h.exp()`) on $2 \times 2$ Hermitian operators (such as Pauli spin combinations and two-state hydrogenic subspaces) instead of evaluating the exact analytical $SU(2) \times U(1)$ closed form in $O(1)$ operations.

## Decision

1. **`core`-Slice Complex Thomas Solver (`pure_math::pure_math::analysis::pde::tridiagonal`)**:
   - Add `factor_tridiagonal`, `solve_tridiagonal_factored`, `solve_tridiagonal_factored_in_place`, `solve_tridiagonal_in_place`, and `apply_tridiagonal` using exclusively `core` slices and `Complex<f64>` with `#[verified]` instrumentation and pivot/finiteness validation prior to division.
2. **`SolverWorkspace` and `CrankNicolson1D` (`domain_physics::physics::quantum`)**:
   - Add `SolverWorkspace` and `CrankNicolson1D` in `crates/domain_physics/src/physics/quantum/tridiagonal.rs` (re-exported from `domain_physics::physics::quantum`), delegating Cayley factorization and sweeps to `pure_math::pure_math::analysis::pde::tridiagonal` and consolidating `CrankNicolsonPropagator1D` onto the same workspace primitives with zero code duplication.
   - Reject non-finite or non-positive `dx`, `dt`, `mass`, and `h_bar` before division so `WaveRunner` and `FiniteStateGuard` record structured diagnostics and recover cleanly.
3. **In-Place Wavepacket, Hamiltonian, and Closed-Form $2 \times 2$ Propagator Helpers**:
   - Add `construct_two_state_eigen_hamiltonian_into` and `gaussian_wavepacket_into` in `hamiltonian.rs` using `core` constants and zero heap allocations over pre-sized buffers.
   - Add `time_evolution_operator_2x2_into` in `schrodinger.rs` (and dispatch $2 \times 2$ Hermitian matrices in `time_evolution_operator` through the same Taylor-stabilized analytical formula) and migrate `HydrogenicSuperposition3D` (`hydrogenic.rs`) and `WaveRunner` (`wave_sim/runner.rs`, `wave_sim/sculpt.rs`) to zero-allocation updates.

## Consequences

- `pure_math::analysis::pde::tridiagonal` provides an allocation-free, `no_std`-compatible $O(N)$ Thomas solver reusable across PDE and physics crates.
- `WaveRunner` parameter updates, `SpatialBrushKit` potential sculpting, wavepacket relaunches at `MAX_GRID_SIZE = 1000`, and `HydrogenicSuperposition3D` basis/propagator reconfigurations execute with zero post-initialization heap allocations under `VerifiedEngine::run_verified`.
- All existing public `domain_physics::physics::quantum` signatures remain intact while $2 \times 2$ Hermitian propagators evaluate in exact $O(1)$ closed form.
