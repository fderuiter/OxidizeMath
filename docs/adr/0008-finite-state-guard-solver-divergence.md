# ADR 0008: Finite State Guard and Interactive Solver Divergence Banner

## Context
In numerical integration loops across interactive tools (`AttractorPlotter` in `attractors.rs` and `OdeSolverTool` in `ode.rs`), divergent parameter configurations or initial conditions previously allowed non-finite float values (`NaN` or `Inf`) to enter state trajectory histories. This resulted in broken visualizations and UI unresponsiveness.

## Decision
We implemented immediate non-finite step interceptors and interactive diagnostic warning banners:
1. Added non-finite state checks in `AttractorPlotter::show` to halt simulation step loops when non-finite states occur, setting `diverged = true` and pausing simulation without pushing `NaN`/`Inf` to trajectory history.
2. Added divergence checks in `OdeSolverTool::recalculate` to halt Runge-Kutta 4 iteration upon encountering non-finite states and show an egui warning banner with parameter reset controls.
3. Added unit tests in both `attractors.rs` and `ode.rs` to verify that divergent parameters trigger the `diverged` flag, halt integration, and prevent non-finite float leakage.
4. Cleaned up duplicate `[package.metadata.cargo-machete]` manifest entries across workspace domain crates to ensure workspace compilation integrity.

## Consequences
- **Safety**: Solver loops halt immediately on divergence, preventing non-finite floats from entering plot data buffers.
- **User Experience**: Clear interactive egui warning banners inform users of divergence and allow easy parameter resets matching MATLAB solver diagnostics.
