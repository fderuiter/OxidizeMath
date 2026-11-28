# ADR 20261127: Tier 2 Shooting Control, Howard Policy Iteration, Mann Mean-Field Games, and Zero-Allocation Nash Equilibrium Solvers in `engine_opt`

## Status

Accepted

## Context

Prior to this decision, constrained optimal control, dynamic programming, mean-field game equilibrium, and finite-game Nash equilibrium algorithms were fragmented or incomplete across Tier 3 domain crates and interactive tools:

1. **Degenerate Viscous Mean-Field Games and Upwind Transport**: `domain_applied::applied::game_theory::mean_field` relied on un-damped fixed-point sweeps that diverged at small congestion viscosity and normalized discrete densities by element sum rather than spatial quadrature $\sum_i m_i \Delta x = 1$. Meanwhile, `engine_pde::StencilOperators` lacked a conservative face-flux operator for spatially varying drift fields.
2. **Heap-Allocating Synchronous Value Iteration**: `domain_ai::ai::reinforcement_learning::bellman_solver::BellmanSolution` allocated seven heap `Vec` buffers per solve and relied on synchronous value iteration, requiring thousands of sweeps near $\gamma = 0.999$.
3. **Missing Constrained Trajectory Optimization and Zero-Allocation Finite-Game Solvers**: Receding-horizon control across `ClosedLoopControlTool`, 4D respiratory radar gating (`domain_physics`), multi-dose pharmacokinetics (`domain_applied`), battery duty scheduling (`domain_applied`), and Almgren-Chriss execution (`pure_math`) lacked a shared `#![no_std]` single-shooting `iLQR` and projected `MPC` engine, while `domain_applied::applied::game_theory::equilibrium` only exposed pure-strategy enumeration without mixed-strategy bimatrix or polymatrix solvers.

## Decision

1. **Conservative Upwind Transport in [`engine_pde`](../../crates/02_engines/engine_pde/README.md)**:
   - Add `StencilOperators::try_new_1d_f64` and `StencilOperators::conservative_upwind_flux` computing discrete conservative flux divergence $-\nabla \cdot (v m)$ for spatially varying face velocities `(v_left, v_right)`.

2. **Expand `#![no_std]` [`engine_opt`](../../crates/02_engines/engine_opt/README.md) with `ScalarWorkspace` Solvers and Typed [`OptError`](../../crates/02_engines/engine_opt/src/error.rs)**:
   - Introduce `#![no_std]` `Copy` `OptError` implementing `math_diagnostics::Diagnostic` and wire `Error::Opt(OptError)` into the [`oxidizemath`](../../crates/04_bindings/oxidizemath/README.md) umbrella facade.
   - Implement `engine_opt::mfg` (`solve_mfg_1d`, `mfg_1d_workspace_len`) per Achdou & Capuzzo-Dolcetta (2010) using monotone Engquist-Osher Hamiltonians, adaptive CFL advection-diffusion substeps down to `viscosity = 0.0`, quadrature normalization `1.0 / (sum * dx)`, and Mann fixed-point damping with $L^1$ residual tracking.
   - Implement single-shooting box-projected `engine_opt::ilqr` (`solve_ilqr`, `ilqr_workspace_len`) and receding-horizon `engine_opt::mpc` (`solve_mpc`, `mpc_workspace_len`, `warm_start_shift_controls`) backed by `engine_linalg::cholesky_solve` Riccati sweeps and Armijo backtracking.
   - Implement `engine_opt::dp` (`solve_infinite_horizon_dp`, `solve_finite_horizon_dp`) using exact Howard policy iteration via `engine_linalg::lu_solve`.
   - Implement `engine_opt::nash` (`solve_bimatrix_lemke_howson`, `solve_polymatrix_regret_matching`) providing lexicographic complementary pivoting and polymatrix `CFR+` regret matching over `ScalarWorkspace`.

3. **Unify Downstream Domain Solvers and Closed-Loop Controllers**:
   - Route `FixedPointSolver` (`domain_applied`) and Braess congestion profiles (`apps/math_explorer_gui/src/tabs/paradox_lab/braess.rs`) through `engine_opt::mfg`.
   - Refactor `BellmanSolution` and `TabularGridPolicy` (`domain_ai`) onto inline stack buffers backed by `solve_infinite_horizon_dp` and `lu_solve`.
   - Add `MpcController` (`apps/math_explorer_gui/src/tabs/ai/closed_loop_control/controllers/mpc.rs`) reusing analytical RK4 Jacobians (`rk4_step_with_jacobian`, `pk_rhs_and_jac`), alongside harmonic radar lookahead (`domain_physics`), pharmacokinetic regimen `MPC` (`domain_applied`), battery duty scheduling (`domain_applied`), Almgren-Chriss execution `MPC` (`pure_math`), and zero-allocation Nash solvers (`domain_applied`).

## Consequences

- **Positive**: Eliminates all seven heap `Vec` allocations from `BellmanSolution`, converging in `< 20` sweeps at $\gamma = 0.999$ (`< 1e-9` Bellman residual), stabilizes zero-viscosity (`viscosity = 0.0`) Mean-Field Games with exact mass conservation, and holds `PkPlantTwin` inside `[pk_mec, pk_mtc]` without cold-start overdoses under `VerifiedEngine::run_verified`.
- **Positive**: Preserves downward tier dependencies (`03_domains -> 02_engines -> 00_foundation`) and `#![no_std]` WebAssembly compatibility across `engine_pde` and `engine_opt`.
- **Trade-offs**: Adds `engine_opt -> engine_pde`, `domain_ai -> engine_linalg / engine_opt`, and `pure_math -> engine_opt` dependency edges recorded in [`system_overview.md`](../architecture/system_overview.md) and [`api_baseline.json`](../../apps/unified_verification/api_baseline.json).
