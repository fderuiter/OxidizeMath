# ADR 20261123: Coupled Chaos Parameter Atlases and Zero-Allocation Lyapunov Kernels

## Status
Accepted (extends [ADR 0012](0012-background-simulation-worker-offload.md), [ADR 0021](0021-inplace-row-softmax-allocation-elimination.md), and [ADR 0044](0044-unify-build-time-traceability-engine.md))

## Context
Three correctness, allocation, and responsiveness gaps affected the chaos domain (`crates/domain_physics/src/physics/chaos/`, `math_explorer/`, and `math_explorer_gui/src/tabs/chaos/`):
1. **Heap Allocations and Singularity Gaps in Chaos Metrics**:
   - `correlation_dimension` in [`crates/domain_physics/src/physics/chaos/fractals.rs`](../../crates/domain_physics/src/physics/chaos/fractals.rs) cloned `trajectory.to_vec()` on every call, violating NASA Power of 10 Rule 3 (`VerifiedAllocator` `MEMORY_LOCK`) on live simulation hot paths.
   - `logistic_lyapunov` in [`crates/domain_physics/src/physics/chaos/metrics.rs`](../../crates/domain_physics/src/physics/chaos/metrics.rs) produced `NaN` for `n == 0` and `-inf` at superstable zero-derivative singularities ($x = 0.5$ or $r = 0$).
   - `lorenz_lyapunov` allocated heap `String` errors when `evolution_time < 0.5 * time_step` or when trajectories contracted to identical floating-point coordinates ($d_t = 0$).
2. **Hardcoded Scalar Lorenz Regime Heuristic**:
   - `LorenzSystem::theory_description` in [`crates/domain_physics/src/physics/chaos/lorenz.rs`](../../crates/domain_physics/src/physics/chaos/lorenz.rs) hardcoded `self.rho > 24.74`, misclassifying decoupled stable systems such as $\sigma = 0, \rho = 28$ and high-$\rho$ periodic windows as `"Chaotic regime"`.
3. **Uncoupled GUI Parameter Controls & UI-Thread Sweeps**:
   - `FractalViewer` reset viewports on Mandelbrot/Julia mode switches and lacked live hover Julia previews and click/`Enter` teleportation; `BifurcationDiagram` lacked a coupled $x_n$ iterate time-series inspector alongside cobweb and Lyapunov curves; and `AttractorPlotter` allocated sorted trajectory clones during correlation invariant updates.

## Decision
1. **Zero-Allocation In-Place Correlation Sum & Finite Lyapunov Kernels**:
   - Add `#[verified_engine::verified] pub fn correlation_dimension_in_place(trajectory: &mut [Vector3<f64>], epsilon: f64) -> f64` in [`crates/domain_physics/src/physics/chaos/fractals.rs`](../../crates/domain_physics/src/physics/chaos/fractals.rs) sorting caller-provided scratch slices via stack-only `sort_unstable_by`, and delegate `correlation_dimension` to it.
   - Guard `n == 0`, non-finite inputs, and zero-derivative/overflow singularities in `logistic_lyapunov`, and handle zero-distance trajectory contraction ($d_t \le 10^{-30}$) and parameter validation without heap `String` allocations in `lorenz_lyapunov`.
   - Classify Lorenz regimes dynamically via `lorenz_lyapunov` and analytical Hopf threshold $\rho_H(\sigma, \beta)$ in `LorenzSystem::theory_description`, including guards for non-positive $\sigma$ or $\beta$.
   - Update [`math_explorer/examples/lorenz_chaos.rs`](../../math_explorer/examples/lorenz_chaos.rs) to report `lorenz_lyapunov` alongside trajectory divergence, and align `TraceabilityEngine::discover_workspace_dirs` in [`crates/oxidize_core/src/traceability.rs`](../../crates/oxidize_core/src/traceability.rs) with `IGNORED_TRACEABILITY_MEMBERS`.
2. **Progressive Coupled Parameter Atlases in `math_explorer_gui`**:
   - Unify 2D regime cartography across `FractalCartographer`, `BifurcationCartographer`, and `AttractorCartographer` via `ParameterAtlasController`.
   - Preserve independent $c$-plane and $z$-plane `(center, zoom)` viewports in `FractalViewer`, adding `linked_explorer` (`#[serde(default)]`) alongside `c_pinned` probing, a live `160x120` hover Julia inset, and progressive teleportation.
   - Offload `BifurcationDiagram` sweeps into `BifurcationRunner` under `BackgroundWorker` (`8ms` cooperative `PlatformInstant::now()` slices) with `logistic_lyapunov` overlays and side-by-side cobweb and $x_n$ iterate time-series inspectors.
   - Reuse pre-allocated scratch capacity with `correlation_dimension_in_place` in `AttractorPlotter` invariants telemetry alongside live `lorenz_lyapunov` badges and 2D $(\sigma, \rho)$ / $(\rho, \beta)$ Hopf regime atlases.

## Consequences
- **Zero-Allocation Safety**: `logistic_lyapunov`, `lorenz_lyapunov`, and `correlation_dimension_in_place` execute with zero dynamic heap allocations under `VerifiedAllocator` (`MEMORY_LOCK`).
- **Mathematical Fidelity**: Lorenz regime descriptions reflect actual dynamical divergence and Hopf thresholds across $(\sigma, \rho, \beta)$, properly classifying $\sigma = 0$ and periodic windows.
- **Responsive Desktop & WebAssembly UX**: All heavy fractal, bifurcation, and 2D parameter atlas sweeps yield within cooperative slice budgets on both native worker threads and single-threaded `wasm32`.
