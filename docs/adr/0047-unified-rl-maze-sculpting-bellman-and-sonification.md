# 0047: Unified RL Maze Sculpting, Exact Bellman Solvers, and TD Surprise Sonification

## Status

Accepted

## Context

The Reinforcement Learning tools inside `AiTab` (`GridWorldTool`, `QTableInspectorTool`, and `RewardPlotsTool`) previously operated on static `GridWorldEnv` trap lists without live spatial obstacle sculpting, stochastic crosswind dynamics, exact dynamic-programming ground truth, policy-gradient rollouts, or Web Audio sonification:

1. `crates/domain_ai` already defined `#[verified]` single-state Bellman backup equations (`bellman_optimality_value`, `state_value_bellman_equation`, `action_value_bellman_equation`) and `policy_gradient_step`, yet no value-iteration or policy-evaluation driver connected them to the interactive GridWorld tools.
2. `SpatialBrushKit` supported `egui_plot` canvases via `sample_plot`, whereas discrete cellular RL grids allocate custom `egui::Rect` canvases via `ui.allocate_painter`.
3. `HashMap<(GridState, Move), f64>` lookups and per-step `Vec<Move>` action allocations prevented zero-allocation inner rollout loops.

## Decision

1. **API-Baseline-Compatible `SculptedGridMdp` and `DenseRlWorkspace` (`domain_ai`)**:
   - Preserve `GridWorldEnv` public fields and `GridWorldEnv::step` untouched to maintain `api_baseline.json` compatibility, and provide `From<&GridWorldEnv> for SculptedGridMdp`.
   - Introduce `GridCellRole` (`Free`, `SolidWall`, `Goal`, `Trap`, `WindUp`, `WindDown`, `WindLeft`, `WindRight`) on `SculptedGridMdp` with bounce-back walls and normalized stochastic wind slip $p_{\text{wind}} \in [0, 1]$ (`\sum_{s'} P(s'|s,a) = 1.0`).
   - Add `DenseRlWorkspace` indexed by `flatten_2d_index(x, y, width)` alongside zero-allocation `step_tabular_q_into` and `step_softmax_pg_into` rollout kernels (plus `compute_softmax_pg_delta` wrapping `policy_gradient_step`).

2. **Exact Bellman Value Iteration and Policy Evaluation (`bellman_solver.rs`)**:
   - Implement `BellmanSolution::solve` and `BellmanSolution::evaluate_policy` (`TabularGridPolicy`) invoking `bellman_optimality_value`, `action_value_bellman_equation`, and `state_value_bellman_equation` over stack-resident reachable state neighborhoods (`[GridState; 9]`), recomputed lazily when grid topology or $\gamma$ changes.

3. **Custom Rect Brush Sampling & RLE Worker Synchronization (`math_explorer_gui`)**:
   - Add `SpatialBrushKit::sample_rect` (`framework/spatial_brush/rect_canvas.rs`) for `egui::Sense::click_and_drag()` grid canvases, sharing the WCAG 2.2 AA keyboard reticle (`Arrow`, `Shift+Arrow`, `[`, `]`, `Space`, `Enter`), `TwinSlotLiveRegion` announcements, and `focus_ring_stroke` indicator without trapping `Tab` or `Shift+Tab`.
   - Synchronize sculpted cell masks and RL topology parameters across `QTableInspectorTool` and `RewardPlotsTool` workers via `SimCommand::ApplyBrushStroke` and `SimCommand::SyncSculptState`, and persist custom masks compactly via `encode_rle_mask` / `decode_rle_mask` within the `4,096`-byte URL budget.

4. **Temporal-Difference Surprise Sonification (`audio/mappings/reinforcement_learning.rs`)**:
   - Add stack-allocated `rl_td_surprise_frame(state_value, td_error, reward, wind_pan)` mapping $V(s_t)$ to carrier pitch (`140..=660 Hz`), $|\delta_t|$ to FM modulation index and filter brightness, crosswind slip to stereo pan, and terminal rewards to consonant/dissonant harmonic transients under `-3.0 dBFS`.

## Consequences

- Users can sculpt walls, traps, goals, and directional crosswinds live by pointer drag or keyboard cursor while comparing empirical Q-learning and REINFORCE policies against exact Bellman $V^*(s)$ and $V_\pi(s)$ fields.
- Both `DenseRlWorkspace` rollouts and `OfflinePcmSynthesizer` rendering of `rl_td_surprise_frame` execute 1,000-step verification runs under `VerifiedAllocator` with `0` heap allocations.
