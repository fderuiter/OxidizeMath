# Architecture Decision Record 0008: Finite State Guard and Interactive Solver Divergence Banner

## Context
Numerical solver loops in `attractors.rs` (`AttractorPlotter`) and `ode.rs` (`OdeSolverTool`) previously appended non-finite floats (`NaN` or `Inf`) directly to plot trajectories when equations diverged. This caused invalid rendering states and unhandled divergence behavior during interactive simulation steps.

## Decision
We implemented immediate divergence guards and interactive warning banners across GUI numerical solvers:
1. Added finite value checks (`f64::is_finite`) in `AttractorPlotter::show` and `OdeSolverTool::recalculate` prior to pushing step results into history/series.
2. When non-finite values are encountered, integration is halted, `diverged` is set to `true`, and non-finite states are excluded from trajectory histories.
3. Interactive egui warning banners with parameter reset triggers (`reset_parameters()`) are displayed when `diverged` is true.
4. Parameter adjustments and reset controls cleanly reset the `diverged` flag.
5. Added unit tests in both modules to verify divergence detection, non-finite value exclusion, and parameter reset behavior.

## Consequences
- **Positive:** Trajectory plots remain free of `NaN` and `Inf` values, preventing rendering glitches or crashes.
- **Positive:** Interactive warning banners notify users of solver divergence with clear recovery actions matching MATLAB solver diagnostics.
- **Positive:** Parameter modifications and resets clear divergence state cleanly.
