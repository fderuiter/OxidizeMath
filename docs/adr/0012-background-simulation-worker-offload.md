# 12. Background Simulation and Reinforcement Learning Worker Offloading

## Status
Accepted

## Context
Long-running numerical simulations (reaction-diffusion systems) and reinforcement learning agent training loops previously executed synchronously on the user interface render thread. This led to UI stalls, frame rate drops during batch training, and unbounded communication buffers between UI and compute loops.

## Decision
1. Added `Send` bound to `ExplorationStrategy` trait and boxed strategy references in `crates/domain_ai` so agents can safely cross worker thread boundaries.
2. Implemented bounded communication channels (`mpsc::sync_channel(4)`) in `SimulationController` with non-blocking `try_send` to decouple worker execution from render frame rates.
3. Added `is_complete(&self) -> bool` method to `SimulationRunner` and `SimCommand::TrainEpisodes(usize)` for auto-completing bounded batch runs.
4. Extracted WebAssembly worker event listener and suspend/wake loop into `async_sim::wasm` maintaining file length constraints.
5. Offloaded morphogenesis reaction-diffusion numerical stepping and RL training loops to background worker threads.
6. Filtered out external tool download directories (`cargo-machete*`, `dist`, `artifacts`) from test discovery in `crates/markdown_tests/build.rs`.

## Consequences
- Heavy mathematical simulation and RL episode stepping run asynchronously without blocking the UI thread.
- Thread-safe exploration strategies allow agent state migration across threads.
- Non-blocking state queues bound memory footprint while running at 60 FPS.
- Pre-flight workspace doc tests are isolated from external tool binaries and extraction artifacts.
