# Architecture Decision Record 0016: ODE Multi-Solver Comparative Trace Overlays

## Status

Accepted

## Context
Interactive numerical ODE tools in Math Explorer previously used a single hardcoded integrator (`RungeKutta4`) without comparative trace visualization. This limitation prevented stability and accuracy benchmarking across different numerical integration algorithms. Computational scientists require direct comparison between first-order (`Euler`), second-order (`Heun`), and fourth-order (`RungeKutta4`) solvers to observe truncation errors, numerical drift, and step-size stability boundaries.

## Decision
We implemented multi-solver comparative trace overlays and quantitative error metrics:
1. Extended `IntegrationMethod` in `crates/pure_math` to support `RungeKutta4`, `Euler`, and `Heun` integrators with display names and collection helpers.
2. Implemented `Heun<State>` (explicit trapezoidal rule / RK2) in `crates/pure_math/src/pure_math/analysis/ode/solvers.rs` with pre-allocated scratch buffers to ensure zero dynamic memory allocation during step loops.
3. Updated domain physics integrations (`LorenzAttractor` and `BlochSimulation`) to support the `Heun` method.
4. Added an integration algorithm selector combo box in `OdeSolverTool` to configure the active solver.
5. Added a multi-solver comparison toggle switch in `OdeSolverTool` that recalculates trajectories across all integrators and renders them as overlaid traces on identical plot axes using `TraceSnapshotManager`.
6. Enforced finite state checks across all solver steps to display immediate divergence warnings when non-finite states occur.
7. Calculated quantitative discrepancy metrics (maximum absolute discrepancy and RMS discrepancy) displayed via copyable metrics in the sidebar summary panel.

## Consequences
- **Positive:** Users can benchmark accuracy and convergence behavior across numerical integrators on identical axes.
- **Positive:** Solver execution preserves zero heap allocation invariants during simulation loops.
- **Positive:** Interactive divergence warnings immediately inform users if step size parameters cause lower-order integrators to diverge.
- **Positive:** Quantitative discrepancy metrics provide numerical confirmation of integrator accuracy ordering.
