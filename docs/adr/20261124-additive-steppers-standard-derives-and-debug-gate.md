# ADR 20261124: Additive Steppers, Standard Derives, and Public `Debug` Gate

## Status

Accepted (2026-11-24; reconciles with [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md))

## Context

Several cross-cutting ergonomic, allocation, and time-propagation gaps existed across the ODE, reaction-diffusion, and domain simulation crates:

1. **Per-step solver allocation and ignored `integration_method`:** [`LorenzSystem`](../../crates/domain_physics/src/physics/chaos/lorenz.rs) and [`BlochSimulator`](../../crates/domain_physics/src/physics/mri/bloch.rs) did not retain pre-allocated ODE solver scratch workspaces across steps and did not dispatch uniformly across configured `IntegrationMethod` variants in both inherent `step` and `TimeStepper::step`. Note that [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md) intentionally deleted obsolete `oxidize_core::{ModelConfig, ModelState, SimulationModel}` traits and dead wrapper structs (`LorenzConfig`, `BlochConfig`, `BlochState`), standardizing all active simulators onto `TimeStepper` and `EvolutionEngine`.
2. **Cloning in `TimeStepper` and hardcoded `t = 0.0`:** [`TimeStepper`](../../crates/pure_math/src/pure_math/analysis/ode/stepper.rs) previously delegated to `SolverExt::solve`, cloning heap-backed state vectors every step, while stateful steppers (`OdeModel`, `SIRModel`, `SEIRModel`, `LorenzSystem`, `BlochSimulator`, `ReactionDiffusionSystem`, and `StandardSolverAdapter`) did not uniformly track and forward elapsed simulation time `t` across trait and inherent stepping.
3. **Per-node heap allocation in `ReactionModel::add_reaction_batch`:** The default batch reaction helper in [`crates/domain_biology/src/biology/reaction_diffusion/traits.rs`](../../crates/domain_biology/src/biology/reaction_diffusion/traits.rs) allocated `Vec<f64>` scratch buffers on every timestep.
4. **Missing standard traits and unguarded `Debug` regressions:** Numerous public types across `math_commons`, `engine_linalg`, `pure_math`, and `domain_*` lacked `Debug`, `Clone`, `PartialEq`, `Copy`, or `Default`, blocking `assert_eq!`, `dbg!`, and `Result::unwrap_err()`, while [`README.md`](../../README.md) omitted `TuringSolverStrategy` and `StandardSolverAdapter` in its Strategy Pattern diagram.

## Decision

We adopt a four-part additive standardization across the workspace while preserving [ADR 0036](0036-delete-simulation-model-traits-and-physics-adapters.md) and maintaining SemVer signature stability in `Certified` crates:

1. **Pre-allocated ODE dispatch and zero-allocation `TimeStepper`:** Introduce [`PreallocatedOdeSolver`](../../crates/pure_math/src/pure_math/analysis/ode/configured_solver.rs) (`Euler`, `Heun`, `RungeKutta4`, `ImplicitMidpoint`, `GaussLegendre4`) and `VectorOperations::take_for_step` so `TimeStepper`, `OdeModel`, `LorenzSystem`, and `BlochSimulator` mutate states in place with zero post-initialization heap allocations under `VerifiedAllocator` and bit-identical trajectories between inherent `step` and `<Self as TimeStepper<_>>::step`.
2. **Elapsed simulation time tracking:** Track `t: f64` (`pub const fn time(&self) -> f64`) across `OdeModel`, `SIRModel`, `SEIRModel`, `LorenzSystem`, `BlochSimulator`, `ReactionDiffusionSystem`, and `StandardSolverAdapter`, and expose additive time-aware methods (`elapsed_time`, `advance_time`, `step_at`, `step_with_at`, `reaction_at_time`, `add_reaction_batch_at_time`).
3. **Stack-buffered reaction batches:** Replace the per-step `Vec<f64>` buffers in `ReactionModel::add_reaction_batch` / `add_reaction_batch_at_time` with fixed stack arrays (`STACK_REACTION_SPECIES_CAP = 16`), falling back to heap vectors only above 16 species.
4. **Workspace-wide `Debug` / value-type derives and `unified_verification` gate:** Implement `Debug` on all reachable public structs and enums across first-party workspace library crates (deriving `Clone`, `PartialEq`, `Copy`, and `Default` on plain-data types and implementing manual `core::fmt::Debug` on closure, trait-object, FFT planner, channel, and AST cache structs), update [`README.md`](../../README.md) to document `TuringSolverStrategy` and `StandardSolverAdapter`, regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json), and enforce public `Debug` coverage in [`crates/unified_verification/src/public_debug.rs`](../../crates/unified_verification/src/public_debug.rs) (`check-public-debug`, `check-api-drift`, and `policy_audit`).

## Consequences

- **Positive:** All stateful ODE and reaction-diffusion steppers achieve zero post-initialization heap allocations in `VerifiedEngine::run_verified` loops, accurately propagate elapsed time `t` for non-autonomous systems while keeping autonomous `Euler`/`Heun`/`RungeKutta4` arithmetic bit-identical, and honor `IntegrationMethod` consistently.
- **Positive:** Every reachable public struct and enum across first-party library crates implements `Debug` (with `Clone` and `PartialEq` on plain-data types), enabling `Result::unwrap_err()`, `dbg!`, and `assert_eq!` ergonomics with automated CI enforcement.
- **Neutral / Trade-offs:** Stateful model structs store an extra `f64` elapsed-time field and pre-allocated solver scratch buffers, trading a small initialization footprint for zero per-step heap traffic.
