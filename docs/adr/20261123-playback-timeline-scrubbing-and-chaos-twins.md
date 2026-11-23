# ADR 20261123: Playback Timeline Scrubbing, Ring Buffers, and Twin Chaotic Trajectories

## Status

Accepted

## Context

`PlaybackTransport` previously exposed forward-only `Play`, `Pause`, `Step`, and `Reset` controls (`Space`, `Period`, `R`). While sufficient for dissipative or monotonic PDE grids, chaotic ODE systems (Lorenz, Rössler, and the planar Lagrangian double pendulum) exhibit sensitive dependence on initial conditions ($\|\delta\mathbf{x}(t)\| \approx \|\delta\mathbf{x}(0)\| e^{\lambda_1 t}$), where macroscopic lobe flips emerge suddenly after many Lyapunov times. Without backward stepping, random-access scrubbing, editable initial conditions, and a synchronized $\log_{10}\|\delta\mathbf{x}(t)\|$ subplot, users could not inspect or rewind across the critical divergence horizon.

At the same time, OxidizeMath enforces strict invariants:

1. Zero dynamic heap allocations inside `VerifiedAllocator`-locked `SimulationRunner::step` and `SimulationRunner::step_once` paths.
2. Identical async simulation behavior across native threads and `wasm32` cooperative main-thread stepping.
3. Compact URL-shareable JSON state budgets (`< 400` bytes for `AttractorPlotter`, well within `URL_JSON_BOUNDS`), coexisting with `SimulationFramework::rollback_step_back` fault recovery and `ParameterAtlasController` 2D parameter sweeps.
4. Non-opted-in `PlaybackTransport` tools must retain unchanged layouts and behavior.

## Decision

1. **Pre-Allocated `TimelineRingBuffer<T>` (`math_explorer_gui/src/framework/timeline_buffer.rs`) & `CheckpointRing` Integration**:
   - Introduced a fixed-capacity (`4,096`-frame) `TimelineRingBuffer<T: Copy>` that allocates its backing `Vec<T>` once on construction and overwrites slots in-place via circular modular indexing.
   - Supports zero-allocation `push`, `step_back`, `seek`, and automatic in-place truncation of future frames when forward stepping resumes after rewinding, while `AttractorRunner` also implements `CheckpointRing` hooks (`checkpoint_stride() -> Some(1)`) so `PlaybackTransport::timeline(Option<TimelineMetadata>)` drives unified backward stepping and scrubbing.
2. **Unified `PlaybackTransport` Time-Travel (`math_explorer_gui/src/framework/playback.rs`)**:
   - Unified onto `PlaybackAction::StepBack` (`Comma` shortcut gated by `is_pane_shortcut_allowed`) and `PlaybackAction::ScrubTo(usize)` via `.timeline(Option<TimelineMetadata>)`.
   - When `timeline` is `None`, existing forward-only tools remain visually and behaviorally unchanged.
3. **Cross-Platform `SimCommand::StepBack(usize)` and `SimCommand::SeekStep(usize)` (`async_sim/`)**:
   - Integrated with `SimulationController::dispatch_playback` and `worker.rs` so both native background workers and `wasm32` frame loops pause cleanly (`SimWorkerState::Ready`), restore or replay checkpoints / `TimelineRingBuffer` frames, and publish an updated `StateSnapshot` (`TimelineMetadata`).
4. **Verified Rössler & Double Pendulum Solvers, Twin Divergence UI, and 2D Parameter Atlas Coexistence**:
   - Implemented `RosslerSystem` and `DoublePendulumSystem` (`DoublePendulumBuilder` / `DoublePendulumState` over `ArrayState<4>` delegating canonical dynamics to `DoublePendulum` with pre-allocated `RungeKutta4` scratch buffers) alongside `double_pendulum_lyapunov` and `finite_time_lyapunov` in `crates/domain_physics/src/physics/chaos/`, re-exported through `math_explorer`.
   - Refactored `AttractorPlotter` into focused `< 500`-line submodules (`runner_types.rs`, `runner.rs`, `runner_checkpoint.rs`, `twin_state.rs`, `ui.rs`, `scrubber_ui.rs`, `divergence_plot.rs`, `atlas.rs`, `preset.rs`) supporting primary + shadow 3D curves, symbolic `MathExpressionInputBar` initial offsets evaluated once outside stepping loops, Wolf $\lambda_1 t / \ln(10)$ reference slope, windowed FTLE, on-demand Grassberger-Procaccia $C(r)$, and `AttractorCartographer` 2D parameter atlas sweeps.
   - Excluded ring-buffer frames from `AttractorSaveState` so serialized preset/URL JSON remains `< 400` bytes.

## Consequences

- Users can pause, press `Comma` or drag either the transport scrubber or the $\log_{10}\|\delta\mathbf{x}(t)\|$ subplot cursor to rewind across chaotic lobe flips, and step forward with bit-identical deterministic RK4 trajectory reproduction.
- `VerifiedEngine::run_verified` confirms zero heap allocations across `Lorenz`, `Rossler`, and `DoublePendulum` twin stepping.
- High-memory 2D PDE grid tools (`LbmUnified`, `MorphogenesisTool`, `WaveUnified`) and non-opted-in tools remain unaffected without memory bloat.
