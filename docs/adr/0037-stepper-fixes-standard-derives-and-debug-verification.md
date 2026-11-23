# ADR 0037: Additive Stepper Fixes, Standard Derives, and Public `Debug` Verification

## Status
Accepted (extends [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md) and [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md))

## Context
Following the removal of dead `oxidize_core::{ModelConfig, ModelState, SimulationModel}` lifecycle traits and unused physics adapters in [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md), several numerical stepping, zero-allocation, and API ergonomics gaps remained across the mathematical crates:
1. **Physics Integrator Configuration & Pre-Allocated Scratch Buffers**: `LorenzBuilder`, `LorenzSystem`, and `BlochSimulator` needed first-class `IntegrationMethod` (`Euler`, `Heun`, `RungeKutta4`) configuration that is honored identically across inherent `step(dt)` and `TimeStepper::step(dt)` using scratch buffers pre-allocated at initialization rather than constructing solvers per step.
2. **In-Place `TimeStepper` & Elapsed Simulation Time `t`**: `TimeStepper` in `crates/pure_math/src/pure_math/analysis/ode/stepper.rs` previously cloned `State` via `SolverExt::solve` and hardcoded `t = 0.0`. Likewise, `OdeModel`, `SIRModel`, `SEIRModel`, `ReactionDiffusionSystem`, `StandardSolverAdapter`, `FluidParticleSystem`, `LorenzSystem`, and `BlochSimulator` evaluated derivatives at `t = 0.0` rather than tracking and advancing elapsed simulation time `t`.
3. **Reaction Batch Heap Allocations**: `ReactionModel::add_reaction_batch` in `crates/domain_biology/src/biology/reaction_diffusion/traits.rs` allocated two `Vec<f64>` scratch buffers on every derivative evaluation, violating the zero-allocation runtime rule (`docs/architecture/zero_allocation_runtime.md`) during `ReactionDiffusionSystem::step`.
4. **Missing Standard Derives & Regression Gate**: More than a third of reachable public structs and enums across the eight mathematical crates (`math_commons`, `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, and `domain_physics`) lacked `Debug` (and plain-data types lacked `Clone` / `PartialEq`), preventing `assert_eq!`, `dbg!`, and `Result::unwrap_err()` in unit and downstream tests. In addition, the Strategy Pattern diagram in `README.md` did not reflect `TuringSolverStrategy` and `StandardSolverAdapter`.

## Decision
1. **Zero-Allocation In-Place `TimeStepper` & Pre-Allocated Physics Steppers**:
   - Add `VectorOperations::take_for_step(&mut self) -> Self` (overridden with zero-capacity `mem::replace` swaps on heap-backed `VecState`, `DVector<f64>`, `ChemicalState`, and `TuringState<N>`) so `TimeStepper::step_with_at` and `TimeStepper::step_with` mutate state in place via `Solver::step` with zero heap allocations.
   - Add `current_time`, `advance_time`, `step_at`, and `step_with_at` default methods to `TimeStepper`.
   - Keep `oxidize_core::{ModelConfig, ModelState, SimulationModel}`, `LorenzConfig`, `BlochConfig`, and `BlochState` deleted per [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md), and expose `IntegrationMethod` selection directly on `LorenzBuilder` (`integration_method`), `LorenzSystem` (`integration_method`, `set_integration_method`, `step_at`, `time`, `set_time`), and `BlochSimulator` (`with_integration_method`, `integration_method`, `set_integration_method`, `step`, `step_at`, `step_with`, `time`, `set_time`), backed by pre-allocated `Euler`, `Heun`, and `RungeKutta4` scratch buffers.
2. **Elapsed Time `t` Tracking & Stack-Buffered Reaction Batches**:
   - Track elapsed time `t` (starting at `0.0`, advancing by `dt` per step, with `time()` and `set_time(t)` accessors) across `OdeModel`, `SIRModel`, `SEIRModel`, `ReactionDiffusionSystem`, `StandardSolverAdapter`, `FluidParticleSystem`, `LorenzSystem`, and `BlochSimulator`, while preserving autonomous `Euler`, `Heun`, and `RungeKutta4` arithmetic bit-for-bit.
   - Add additive `reaction_at` and `add_reaction_batch_at` default methods on `ReactionModel` (and `reaction_at` on `ReactionKinetics<N>`), forwarding `t` from `ReactionDiffusionModel::derivative_in_place` and `TuringDynamics::derivative_in_place`, and replace heap allocations in `ReactionModel::add_reaction_batch_at` with stack-allocated `[0.0_f64; 32]` buffers for `n_species <= 32`.
3. **Workspace-Wide Standard Derives & Manual `Debug`**:
   - Add `Debug` (plus `Clone`, `PartialEq`, `Copy`, `Eq`, and `Default` where valid) across all plain-data public structs and enums in the eight mathematical crates, using scratch-ignoring `PartialEq` implementations on double-buffered or solver-buffered simulators (`LorenzSystem`, `BlochSimulator`, `TuringSystem`).
   - Implement `std::fmt::Debug` manually using `debug_struct(...).finish_non_exhaustive()` on the 12 closure, trait-object, and FFT planner types (`TrainingLoop`, `QLearningAgent`, `ScoringContext`, `UnifiedFavoritismModel`, `BestResponseCorrespondence`, `WinRatioAnalysis`, `PotentialFlowField`, `Fft2dWorkspace`, `DirichletBoundary`, `NeumannBoundary`, `FastFourierTransform`, and `RiemannianMetric`).
4. **Verification Enforcement, API Baseline & Documentation**:
   - Enforce 100% `Debug` coverage on reachable public structs and enums across the eight mathematical crates in `unified_verification` (`api_drift::public_debug`, `check-api-drift`, `test-policy-audit`, and standalone `check-public-debug` / `xtask check-public-debug`) sharing `AstCache` without extra file parses.
   - Update `README.md` to accurately depict `TuringSolverStrategy`, `StandardSolverAdapter`, `FusedEulerSolver`, and `TuringDynamics`, and regenerate `crates/unified_verification/api_baseline.json`.

## Consequences
- **Positive**:
  - `LorenzSystem::step`, `BlochSimulator::step`, `FluidParticleSystem`, `TimeStepper::step_with`, `ReactionDiffusionSystem::step`, and `TuringSystem` execute in place with zero post-initialization heap allocations verified under `VerifiedAllocator`.
  - Non-autonomous differential equations and time-dependent reaction kinetics receive accurate elapsed simulation time `t` on every substep while autonomous trajectories remain bit-identical.
  - All 453 reachable public structs and enums across the eight mathematical crates implement `Debug` (with `Clone` and `PartialEq` on plain-data types), enforced continuously by `unified_verification`.
- **Negative**:
  - Pre-allocating `Euler`, `Heun`, and `RungeKutta4` `Vector3<f64>` scratch buffers increases stack size of `LorenzSystem` and `BlochSimulator` by seven `Vector3<f64>` values (168 bytes), enabling zero-allocation runtime switching between integrators.
