# ADR 0042: Playback Timeline Scrubbing, Ring Buffers, and Twin Chaotic Trajectories

## Status

Accepted

## Context

`PlaybackTransport` previously exposed forward-only `Play`, `Pause`, `Step`, and `Reset` controls (`Space`, `Period`, `R`). While sufficient for dissipative or monotonic PDE grids, chaotic ODE systems (Lorenz, Rössler, and the planar Lagrangian double pendulum) exhibit sensitive dependence on initial conditions ($\|\delta\mathbf{x}(t)\| \approx \|\delta\mathbf{x}(0)\| e^{\lambda_1 t}$), where macroscopic lobe flips emerge suddenly after many Lyapunov times. Without backward stepping, random-access scrubbing, editable initial conditions, and a synchronized $\log_{10}\|\delta\mathbf{x}(t)\|$ subplot, users could not inspect or rewind across the critical divergence horizon.

At the same time, OxidizeMath enforces strict invariants:

1. Zero dynamic heap allocations inside `VerifiedAllocator`-locked `SimulationRunner::step` and `SimulationRunner::step_once` paths.
2. Identical async simulation behavior across native threads and `wasm32` cooperative main-thread stepping.
3. Compact URL-shareable JSON state budgets (`< 400` bytes for `AttractorPlotter`, well within `URL_JSON_BOUNDS`), coexisting with `SimulationFramework::rollback_step_back` fault recovery.
4. Non-opted-in `PlaybackTransport` tools must retain unchanged layouts and behavior.

## Decision

1. **Pre-Allocated `TimelineRingBuffer<T>` (`math_explorer_gui/src/framework/timeline_buffer.rs`)**:
   - Introduced a fixed-capacity (`4,096`-frame) `TimelineRingBuffer<T: Copy>` that allocates its backing `Vec<T>` once on construction and overwrites slots in-place via circular modular indexing.
   - Supports zero-allocation `push`, `step_back`, `seek`, and automatic in-place truncation of future frames when forward stepping resumes after rewinding.
2. **Opt-In `PlaybackTransport` Scrubbing (`math_explorer_gui/src/framework/playback.rs`)**:
   - Extended `PlaybackAction` with `StepBack` and `Scrub(usize)` and added opt-in builder methods `.can_step_back(bool)` (`Comma` shortcut gated by `is_pane_shortcut_allowed`) and `.timeline_scrubber(current, len)` (rendering a scrubber slider when `len > 1`).
   - Both controls default to disabled/hidden so existing callers remain visually and behaviorally unchanged.
3. **Cross-Platform `SimCommand::StepBack` and `SimCommand::SeekFrame` (`async_sim/`)**:
   - Extended `SimCommand`, `SimulationController`, and `worker.rs` so both native background workers and `wasm32` frame loops transition to `SimWorkerState::Ready`, restore the requested `TimelineRingBuffer` frame on the runner, and publish an updated `StateSnapshot`.
4. **Verified Rössler & Double Pendulum Solvers and Twin Divergence UI**:
   - Implemented `RosslerSystem` and `DoublePendulum` (over `ArrayState<4>` with pre-allocated `RungeKutta4` scratch buffers) alongside `double_pendulum_lyapunov` and `finite_time_lyapunov` in `crates/domain_physics/src/physics/chaos/`, re-exported through `math_explorer`.
   - Refactored `AttractorPlotter` into focused `< 500`-line submodules (`runner_types.rs`, `runner.rs`, `twin_state.rs`, `ui.rs`, `scrubber_ui.rs`, `divergence_plot.rs`, `preset.rs`) supporting primary + shadow 3D curves, symbolic `MathExpressionInputBar` initial offsets evaluated once outside stepping loops, Wolf $\lambda_1 t / \ln(10)$ reference slope, windowed FTLE, and on-demand Grassberger-Procaccia $C(r)$.
   - Excluded ring-buffer frames from `AttractorSaveState` so serialized preset/URL JSON remains `< 400` bytes.

## Consequences

- Users can pause, press `Comma` or drag either the transport scrubber or the $\log_{10}\|\delta\mathbf{x}(t)\|$ subplot cursor to rewind across chaotic lobe flips, and step forward with bit-identical deterministic RK4 trajectory reproduction.
- `VerifiedEngine::run_verified` confirms zero heap allocations across `Lorenz`, `Rossler`, and `DoublePendulum` twin stepping.
- High-memory 2D PDE grid tools (`LbmUnified`, `MorphogenesisTool`, `WaveUnified`) remain unaffected without memory bloat.
