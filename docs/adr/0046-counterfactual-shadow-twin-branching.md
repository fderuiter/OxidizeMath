# Architecture Decision Record 0046: Counterfactual Shadow-Twin Branching (`CounterfactualBranchEngine`)

## Status

Accepted

## Context
Computational scientists, epidemiologists, MRI physicists, and developmental biologists regularly need to evaluate **mid-flight causal interventions**—forking a live simulation at step $t_k > 0$ ($k \ge 1$) to ask *"what if an infinitesimal state perturbation $\delta x$ or parameter intervention $\theta \to \theta'$ occurred right now?"* while keeping the unperturbed baseline evolving synchronously alongside the active trajectory.

Existing comparative visualization mechanisms in OxidizeMath address orthogonal workflows:
1. **[ADR 0016 (ODE Multi-Solver Comparative Trace Overlays)](0016-ode-multi-solver-comparative-overlays.md)** benchmarks numerical integration accuracy (`Euler`, `Heun`, `RungeKutta4`) by re-integrating identical equations and parameters from $t = 0$. It measures numerical truncation error across solvers rather than causal divergence from a mid-flight state/parameter intervention at $t_k > 0$.
2. **`TraceSnapshotManager`** captures static historical curves after completed runs, so pinned snapshots do not advance in lockstep when the user clicks `Play` or `Step` after $t_k$.

Supporting live causal shadow twins across native threads and single-threaded `wasm32` requires strict adherence to zero-allocation solver step invariants (`VerifiedAllocator`), bounded `wasm32` UI frame budgets ([ADR 0033](0033-wasm32-browser-tests-and-playwright-ci-smoke-gate.md)), compact URL state envelopes (`\le 4\text{ KiB}`), and non-panicking `FiniteStateGuard` recovery ([ADR 0008](0008-finite-state-guard-solver-divergence.md)).

## Decision
We introduced `CounterfactualBranchEngine` in `math_explorer_gui/src/framework/counterfactual/` paired with `TwinSliceBudget` in `math_explorer_gui/src/async_sim/twin.rs` and integrated lockstep shadow twins into four pilot runners (`AttractorPlotter`, `SirTool`, `MriTool`, and `MorphogenesisTool`):

1. **Mid-Flight State, Parameter, and PRNG Cloning at $t_k > 0$:**
   - Triggering `"⑂ Fork Branch Here"` at step $k > 0$ clones `(t_k, x(t_k), rng)` into frozen baseline `Branch A` (retaining parameters $\theta$) while active `Branch B` applies live parameter edits $\theta'$ or sanitized state perturbations $\delta x \in \{0.0\} \cup [10^{-8}, 1.0]$.
   - `"⇄ Swap"` exchanges `Branch A` and `Branch B` states, histories, and parameter sets in place; `"✕ Clear Fork"` drops `Branch A` while preserving `Branch B`; `Reset` clears the fork and re-seeds deterministic state cleanly.

2. **Contrast with $t = 0$ Multi-Solver Benchmarking (ADR 0016):**

```text
ADR 0016 (Multi-Solver Benchmarking)          ADR 0046 (Counterfactual Shadow Twins)
------------------------------------          --------------------------------------
Anchor:        Initial condition at t = 0     Mid-flight causal state at t_k > 0
Fixed:         Parameters θ & state x(0)      Solver algorithm & step size Δt
Varied:        Integrator (Euler/Heun/RK4)    Intervention θ -> θ', δx, or 180° pulse
Stepping:      Batch recomputation from 0     Lockstep Play/Step from t_k onward
Primary Metric: Max / RMS truncation error    ||x_B - x_A||_2, ln||δx||/Δt, ΔI_peak, ΔM_xy
```

3. **Zero-Allocation Lockstep Stepping & `4ms + 4ms` `wasm32` Frame Slicing:**
   - Twin scratch buffers (`RungeKutta4`, `LorenzSystem`, `SirTwinStepper`, and double-buffered `TuringSystem` grids) and ring-buffer capacities are pre-allocated when the fork is created, preserving zero heap allocations inside `VerifiedEngine::run_verified` step loops.
   - On `wasm32-unknown-unknown`, `TwinSliceBudget` partitions the `8ms` per-frame worker budget into `4ms` for `Branch A` and `4ms` for `Branch B` while matching exact tick counts per frame so neither twin drifts ahead nor stalls the browser main thread.

4. **Shared Canvas Overlays, Spatial Heatmaps, and Copyable Divergence Strip:**
   - 2D and 3D trajectory tools render `Branch B` as a solid primary curve, `Branch A` as a WCAG-compliant dashed ghost curve, and a `"Fork @ t_k"` vertical marker/callout.
   - `MorphogenesisTool` provides `Active u_B`, `Split u_A | u_B`, and `Difference |u_B - u_A|` spatial heatmaps across `Dark`, `Light`, and `HighContrast` themes.
   - `render_divergence_strip` surfaces live copyable `||x_B(t) - x_A(t)||_2`, finite-time logarithmic separation slope `ln||δx|| / Δt`, peak separation, and domain intervention deltas (`ΔI_peak` infections averted in `SirTool`, `ΔM_xy` spin-echo recovery gain in `MriTool`, and spatial $L_2 / L_\infty$ norms in `MorphogenesisTool`).

5. **Compact Serialization & Twin Exports:**
   - Presets and share URLs serialize compact fork descriptors (`fork_step`, `post_fork_steps`, baseline $\theta$, and $\delta x$) rather than raw state arrays, keeping all four starter counterfactual presets (`lorenz_butterfly_perturbation`, `sir_lockdown_timing`, `mri_spin_echo_refocusing`, `morphogenesis_inhibitor_fork`) well within `ToolStateBounds::DEFAULT` and the `4,096`-byte URL budget.
   - CSV and PNG exports include labeled `Branch B` and `Branch A` trajectories or twin difference fields whenever a fork is active.

## Consequences
- **Positive:** Researchers can quantify causal interventions (lockdown timing, butterfly sensitivity, Hahn $180^\circ$ spin-echo refocusing, and Turing diffusion shifts) directly on the live canvas with zero-perturbation bit-exact parity (`||x_B - x_A||_2 == 0.0` when $\delta x = 0, \theta' = \theta$).
- **Positive:** Zero-allocation `step()` guarantees, `FiniteStateGuard` rollback synchronization, and `wasm32` `8ms` frame-slice responsiveness remain intact across all four pilot domains.
- **Trade-off:** Active twin mode executes two solver steps per simulation tick, halving per-branch throughput within the fixed `wasm32` `8ms` slice when computationally heavy 2D grids are forked.
