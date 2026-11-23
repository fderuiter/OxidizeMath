# ADR 0041: Planar ODE AST Evaluation, Poincaré Stability Maps, AST Caching, and Recovery-Based URL Sharing

## Status
Accepted (extends [ADR 0008](0008-finite-state-guard-solver-divergence.md), [ADR 0016](0016-ode-multi-solver-comparative-overlays.md), [ADR 0017](0017-native-ast-expression-parser-and-live-input.md), [ADR 0022](0022-unify-preset-state-serialization.md), and [ADR 0029](0029-consolidate-ast-evaluators-and-parser-helpers.md))

## Context
1. **Planar Dynamical Systems & Explicit Step Time**: `crates/math_commons/src/ast/eval.rs` exposed `Expr::eval_2d(t, y, params)` for scalar first-order ODEs, while coupled 2D planar systems $x' = f(t, x, y),\ y' = g(t, x, y)$ require zero-allocation evaluation across $(t, x, y)$. In addition, `OdeModel::step` in `crates/pure_math/src/pure_math/analysis/ode/model.rs` passed static `0.0` as `t` to `Solver::step`, freezing non-autonomous forcing terms such as `y' = -k * y * sin(t)` at $t \in [0, \Delta t]$.
2. **Phase Portraits & Poincaré Bifurcations**: `OdeSolverTool` locked its canvas to 1D time-series plots without vector-field quivers, nullclines, click-to-seed initial conditions, or interactive $(\tau, \Delta)$ trace–determinant stability classification.
3. **Mid-Keystroke Syntax Feedback vs Numerical Divergence**: Incomplete expressions during live typing (such as `cos(`) caused `OdeSolverTool` to set `diverged = true` and trigger the `FiniteStateGuard` numerical divergence banner intended strictly for `NaN`/`Inf` solver blow-up.
4. **URL Payload Budget**: `SimulationFramework::export_url_preset_json` serialized `tool.save_state()`, embedding `TraceSnapshotManager` history runs and exceeding `MAX_DECODED_URL_STATE_BYTES = 4096` after parameter edits.

## Decision
1. **Zero-Allocation Planar AST Evaluator**: Added `Expr::eval_planar(&self, t: f64, x: f64, y: f64, params: &HashMap<String, f64>) -> f64` in `crates/math_commons/src/ast/eval.rs` delegating to `Expr::eval_fast`, verified for zero heap allocations under `VerifiedAllocator`.
2. **Explicit Time Advancement & Stack-Allocated ODE Stepping**: Updated `OdeModel` in `crates/pure_math/src/pure_math/analysis/ode/model.rs` to track and advance explicit time `t` across `TimeStepper::step`, and migrated `OdeSolverTool` solvers in `math_explorer_gui/src/tabs/analysis/ode/` to step stack-allocated `ArrayState<1>` and `ArrayState<2>` states directly via `Solver::step` at explicit $t_i = i \cdot \Delta t$.
3. **2D Phase Plane & Poincaré $(\tau, \Delta)$ Submodules**: Extracted planar quiver rendering (`egui_plot::Arrows` capped at $25 \times 25$), marching-squares nullclines, click-to-seed orbits (capped at 8 within `MAX_ODE_STEPS`), synchronized $x(t)$/$y(t)$ twin timelines, and an interactive $(\tau, \Delta)$ Poincaré minimap ($A = [[0, 1], [-\Delta, \tau]]$) into `ode/phase_plane.rs` and `ode/poincare_map.rs`.
4. **Last-Valid AST Caching**: Extended `MathExpressionInputBar` with `last_valid_expr` and `cached_expr()` so live typing errors display inline syntax feedback while retaining the last valid trajectory without setting `diverged = true`.
5. **Parameter-Only Share URLs**: Switched `SimulationFramework::export_url_preset_json` to `tool.save_recovery_state()`, omitting `TraceSnapshotManager` history from share URLs while preserving full snapshot state in `.json` preset exports and `.mws` workspace files.

## Consequences
- **Positive**: Enables interactive 2D phase-space exploration, equilibrium bifurcation sweeping, and smooth live formula editing without heap allocations in solver inner loops or false divergence banners.
- **Positive**: Shareable simulation URLs remain compact below 4,096 decoded bytes across repeated edits while preserving lossless parameter, view, formula, and seeded-orbit round-trips.
