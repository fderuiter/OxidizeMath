# ADR 0035: Zero-Allocation 2D Symplectic Momentum Solvers and Cross-Domain Loss Landscape Arena

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
1. **Heap-Bound First-Order Optimization Steppers**: [`crates/pure_math/src/pure_math/analysis/optimization.rs`](../../crates/pure_math/src/pure_math/analysis/optimization.rs) implemented [`EvolutionEngine`](../../crates/pure_math/src/pure_math/analysis/evolution.rs) for `SGD` and `Adam` exclusively over heap-allocated `DMatrix<T>` and `DVector<T>` buffers while ignoring the injected `RngCore` parameter. No stack-allocated 2D symplectic Hamiltonian or stochastic Langevin integrators existed in `pure_math` to model momentum slingshots or thermal barrier hopping under `VerifiedAllocator`.
2. **Uncoupled Cross-Domain Physics Primitives**: [`QuadraticHamiltonian`](../../crates/domain_applied/src/applied/game_theory/mean_field/physics.rs) in `domain_applied` and [`LorenzSystem`](../../crates/domain_physics/src/physics/chaos/lorenz.rs) in `domain_physics` were isolated from AI optimization visualizers, while [`LossLandscapeTool`](../../math_explorer_gui/src/tabs/ai/loss_landscape/mod.rs) rendered a static 5-point 2D MSE scatter grid without 3D elevation, particle trajectories, or `InteractiveTool::on_brush` barrier sculpting.

## Decision
1. **Zero-Allocation 2D Momentum & Optimizer Engines in `pure_math`**:
   - Add [`crates/pure_math/src/pure_math/analysis/momentum_dynamics.rs`](../../crates/pure_math/src/pure_math/analysis/momentum_dynamics.rs) providing `PhasePoint2D`, conformal symplectic `HamiltonianLeapfrog2D` (exact Störmer-Verlet at $\gamma = 0$), and stochastic `LangevinBaoab2D` (BAOAB splitting with `rand_distr::Normal` fluctuation-dissipation thermostatting) implementing `EvolutionEngine` with clamped physical bounds ($m \in [0.05, 50.0]$, $\gamma \in [0.0, 20.0]$, $k_B T \in [0.0, 50.0]$, $\Delta t \in [10^{-4}, 0.1]$) and inelastic boundary reflection.
   - Extend `SGD<T>` and `Adam<T, Key>` in [`crates/pure_math/src/pure_math/analysis/optimization.rs`](../../crates/pure_math/src/pure_math/analysis/optimization.rs) with stack `Vector2<T>` `EvolutionEngine` implementations and `AdamState2D<T>` while keeping all `Optimizer` trait signatures unchanged.
2. **Feature-Safe Cross-Domain Coupling**:
   - Enable `"applied"` from the `"ai"` feature in [`math_explorer/Cargo.toml`](../../math_explorer/Cargo.toml) and [`math_explorer_gui/Cargo.toml`](../../math_explorer_gui/Cargo.toml) so `LossLandscapeTool` couples `HamiltonianLeapfrog2D` and `LangevinBaoab2D` with `QuadraticHamiltonian`.
   - Gate `LorenzSystem` chaotic forcing behind `#[cfg(feature = "physics")]` so isolated `--no-default-features --features ai` builds pass `xtask test-features` and gracefully hide the `LorenzChaotic` racer when `physics` is absent.
3. **3D & 2D Sculptable `LossLandscapeTool` Arena**:
   - Decompose `loss_landscape` into [`mod.rs`](../../math_explorer_gui/src/tabs/ai/loss_landscape/mod.rs), [`surfaces.rs`](../../math_explorer_gui/src/tabs/ai/loss_landscape/surfaces.rs), [`arena.rs`](../../math_explorer_gui/src/tabs/ai/loss_landscape/arena.rs), and [`ui.rs`](../../math_explorer_gui/src/tabs/ai/loss_landscape/ui.rs) (`<= 500` lines each), combining `Plot3DViewport` + `Camera3D`, a 2D contour view, `PlaybackTransport`, `on_brush` 32-bump Gaussian sculpting, velocity slingshots, and compact URL-shareable state serialization.

## Consequences
- **Positive**: All 2D optimizer and physical/thermal momentum steppers execute with zero heap allocations under `VerifiedAllocator`, deterministic `OxidizeRng` seeding across native and `wasm32`, and full compatibility with existing `Optimizer`, `DragValue`, export, and URL-sharing contracts.
- **Neutral**: Public API additions in `pure_math::analysis::{momentum_dynamics, optimization}` are recorded in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) via `xtask regenerate-baseline`.
