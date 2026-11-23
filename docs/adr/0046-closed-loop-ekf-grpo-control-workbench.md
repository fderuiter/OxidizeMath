# ADR 0046: Zero-Allocation Discretized Q-Learning Helpers and Unified Closed-Loop Control Workbench

## Status

Accepted (2026-11-23)

## Context

1. **GridWorld Lock-In and Heap Allocation in `QLearningAgent::select_action`:** Prior reinforcement learning tools in [`math_explorer_gui/src/tabs/ai/`](../../math_explorer_gui/src/tabs/ai/mod.rs) coupled `TabularQAgent` exclusively to a static $5 \times 5$ `GridWorldEnv`, while `QLearningAgent::select_action` in [`crates/domain_ai/src/ai/reinforcement_learning/algorithms.rs`](../../crates/domain_ai/src/ai/reinforcement_learning/algorithms.rs) allocated a temporary `Vec<f64>` on every action selection and `TabularQFunction` stored Q-values in a heap `HashMap`.
2. **Uncoupled Nonlinear `KalmanSystem`, `GRPO`, `LorenzSystem`, `SEIRModel`, and `BatemanModel` Primitives:** Nonlinear Extended Kalman Filter (`KalmanSystem`) observers in [`crates/domain_applied/src/applied/algorithms/kalman.rs`](../../crates/domain_applied/src/applied/algorithms/kalman.rs), zero-allocation `GRPO` objectives (`normalize_group_advantages`, `clipped_surrogate_objective`) in [`crates/domain_applied/src/applied/grpo/formulas.rs`](../../crates/domain_applied/src/applied/grpo/formulas.rs), `LorenzSystem` in [`crates/domain_physics/src/physics/chaos/lorenz.rs`](../../crates/domain_physics/src/physics/chaos/lorenz.rs), `SEIRModel` in [`crates/domain_epidemiology/src/epidemiology/compartmental/seir.rs`](../../crates/domain_epidemiology/src/epidemiology/compartmental/seir.rs), and `BatemanModel` / `SuperpositionModel` in [`crates/domain_applied/src/applied/pharmacokinetics/mod.rs`](../../crates/domain_applied/src/applied/pharmacokinetics/mod.rs) lacked a unified closed-loop observer-controller GUI workbench.

## Decision

1. **Zero-Allocation Discretized Q-Learning Storage and Scratch Action Selection (`crates/domain_ai`):**
   - Introduce `DiscreteState`, `DiscreteAction`, and const-generic `FixedArrayQFunction<const STATES: usize, const ACTIONS: usize>` in [`crates/domain_ai/src/ai/reinforcement_learning/q_function.rs`](../../crates/domain_ai/src/ai/reinforcement_learning/q_function.rs) backed by contiguous `[[f64; ACTIONS]; STATES]` arrays with zero dynamic allocations.
   - Add `QLearningAgent::select_action_into` and `QLearningAgent::set_learning_params` in [`crates/domain_ai/src/ai/reinforcement_learning/algorithms.rs`](../../crates/domain_ai/src/ai/reinforcement_learning/algorithms.rs) so hot rollouts evaluate candidate Q-values into a caller-provided `&mut [f64]` scratch slice under `VerifiedAllocator`.

2. **Nonlinear `KalmanSystem` Startup Validation & Stack-Resident `FastEkf` Mirrors (`math_explorer_gui`):**
   - Implement nonlinear `KalmanSystem<f64>` models (`LorenzKalmanModel`, `SeirKalmanModel`, `PkKalmanModel`) using a shared 4-stage RK4 tangent-linear Jacobian kernel validated via `KalmanFilter::builder` at startup and mirrored onto stack-resident `FastEkf<const N: usize>` matrices during locked playback and rollouts.

3. **Unified `ClosedLoopControlTool` in `AiTab`:**
   - Enable `math_explorer/epidemiology` under the `ai` feature in [`math_explorer_gui/Cargo.toml`](../../math_explorer_gui/Cargo.toml) and register `ClosedLoopControlTool` in [`math_explorer_gui/src/tabs/ai/mod.rs`](../../math_explorer_gui/src/tabs/ai/mod.rs) with three split-pane-ready `StarterPresetMetadata` cards pairing with `ChaosTab`, `EpidemiologyTab`, and `ClinicalTrialsTab`.
   - Couple EKF state estimates to critic-free `GRPO` Gaussian policies and discretized `QLearningAgent` controllers with seeded `OxidizeRng` noise/exploration, cooperative `SimulationRunner` episodic training slices (`get_steps_per_frame() == 4`), canvas click-to-kick perturbations, bounded URL sharing, and CSV/image plot export.

## Consequences

- **Positive:** Closed-loop `EKF` + `GRPO` / `QLearningAgent` stabilization runs across chaos, epidemiology, and pharmacokinetic dosing with zero post-startup heap allocations under `VerifiedAllocator`, deterministic `OxidizeRng` replay, and `< 8 ms` cooperative WebAssembly slices.
- **Trade-offs:** Discretized `QLearningAgent` control quantizes continuous EKF features onto a fixed $64 \times 7$ state-action grid to guarantee bounded stack memory and fast tabular convergence alongside continuous `GRPO` policies.

## Verification Commands

```bash
# Regenerate public API baseline and run workspace tests
cargo run -p xtask -- regenerate-baseline
cargo test --workspace --lib --tests --all-features
cargo test --doc --workspace --all-features

# Verify isolated domain features, WASM checks, and full lint gates
cargo run -p xtask -- test-features ai
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
cargo run -p xtask -- lint-all
```
