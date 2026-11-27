# ADR 20261127: Harden AI Solvers, RNG Stream Scaling, and Decision-Boundary Views in Place

## Status

Accepted

## Context

Auditing the optimization, deep-learning, self-calibration, reinforcement-learning, and AI workbench layers surfaced seven correctness and visual-fidelity defects within existing crates:

1. **Optimizer Step-Size and Shape Guards**: [`SGD`](../../crates/03_domains/pure_math/src/pure_math/analysis/optimization.rs) hardcoded `0.01` inside `update_vector` and `update_matrix` instead of applying `self.learning_rate`, silently accepted non-finite or non-positive learning rates, and silently truncated or ignored mismatched tensor shapes.
2. **Degenerate Loss and Softmax Kernels**: [`mean_squared_error`](../../crates/03_domains/domain_ai/src/ai/optimization.rs) and `mse_prime` did not reject empty or `NaN` slices, [`softmax_into`](../../crates/03_domains/domain_ai/src/ai/activation_kernels.rs) collapsed `+Inf` or index-`0` `NaN` logits, and `cross_entropy_softmax_prime` omitted the $\sum_j y_j$ scaling factor required when target vectors are unnormalized.
3. **Identical Per-Layer RNG Streams and Xavier-on-ReLU Variance Collapse**: [`DenseLayer`](../../crates/03_domains/domain_ai/src/ai/deep_learning_theory/linear_algebra.rs), [`TwoLayerMLP`](../../crates/03_domains/domain_ai/src/ai/deep_learning_theory/model.rs), and [`ConvLayer`](../../crates/03_domains/domain_climate/src/climate/autoencoder.rs) reset fresh default `OxidizeRng` streams per layer with symmetric Xavier scaling, duplicating weights across equal-shaped layers and halving ReLU pre-activation variance per layer.
4. **Missing Decision-Boundary Surface in `TrainingMonitorTool`**: [`TrainingMonitorTool`](../../apps/math_explorer_gui/src/tabs/ai/training_monitor.rs) plotted raw spiral points without evaluating `TrainingLoop::predict` across the 2D input plane or exposing optimizer and weight-initialization selectors.
5. **Unnormalized Self-Calibration Clusters and Linear-Space Softmax Targets**: [`calculate_consistency_scores`](../../crates/03_domains/domain_ai/src/ai/self_calibration/scoring.rs) could exceed `1.0` when duplicate answers shared normalized text, and [`calculate_kl_divergence_loss`](../../crates/03_domains/domain_ai/src/ai/self_calibration/training.rs) assigned non-negligible target mass to `0.0`-probability answers and silently accepted degenerate all-zero or `NaN` distributions.
6. **Mid-Episode `REINFORCE` Mutation, Unbootstrapped Timeouts, and Fabricated Q-Tables**: [`rollouts.rs`](../../crates/03_domains/domain_ai/src/ai/reinforcement_learning/rollouts.rs) mutated `theta_logits` and `v_critic` mid-backward-loop without $\gamma^t$ discounting, treated step-limit truncations as `0.0`-value absorbing states, and overwrote `ws.q_table` with critic values.
7. **Deterministic Policy Sampling, Under-Swept Bellman Grids, and Dropped `PolicyGradient` Snapshots**: [`TabularGridPolicy::sample`](../../crates/03_domains/domain_ai/src/ai/reinforcement_learning/bellman_solver.rs) returned deterministic `argmax`, [`EpsilonGreedy::select_action`](../../crates/03_domains/domain_ai/src/ai/reinforcement_learning/strategies.rs) could index past `available_actions`, fixed low sweep counts left residual errors on `20x20` high-$\gamma$ grids, [`QTableSnapshot`](../../apps/math_explorer_gui/src/tabs/ai/q_table_inspector/runner.rs) dropped `theta_logits` during live sculpting, and [`GridWorldTool`](../../apps/math_explorer_gui/src/tabs/ai/grid_world.rs) stalled on step-limit timeouts.

## Decision

1. **Enforce Exact `SGD` Learning-Rate Scaling and Typed Shape/Finiteness Checks**:
   - Apply `self.learning_rate` in `SGD::update_vector` and `SGD::update_matrix`, reject non-finite or non-positive learning rates and non-finite parameter/gradient slices with `Err(OptimizationError::InvalidInput)`, and return `Err(OptimizationError::DimensionMismatch)` on mismatched shapes across `SGD` and `Adam`.
   - Update [`opt_sgd.md`](../math_specs/snippets/opt_sgd.md) and [`pure_math.md`](../math_specs/pure_math.md) to reflect exact step scaling.

2. **Harden `domain_ai` Softmax, MSE, and Self-Calibration Kernels**:
   - Guard empty, mismatched, and `NaN` slices in `mean_squared_error` and `mse_prime`, distribute uniform `1 / k` mass across `+Inf` logits while rejecting `NaN` at any index in `softmax_into`, and scale `cross_entropy_softmax_prime` by $\sum_j y_j$.
   - Normalize self-calibration cluster scores by total probability mass into `[0.0, 1.0]`, compute log-space temperature-scaled softmax targets $P_i \propto \exp(\ln(\max(S_i, 10^{-24})) / \tau)$, and validate finite simplex inputs in `calculate_kl_divergence_loss`.

3. **Adopt Distinct `OxidizeRng` Streams and He et al. (2015) Variance Scaling**:
   - Initialize `DenseLayer`, `TwoLayerMLP`, and `ConvLayer` / `Encoder` / `Decoder` / `Autoencoder` / `Predictor` on distinct `OxidizeRng::split_stream` streams with He variance scaling ($\mathcal{U}(-\sqrt{6 / ((1 + \alpha^2) d_{\text{in}})}, \sqrt{6 / ((1 + \alpha^2) d_{\text{in}})})$ and zero bias), while providing `DenseLayer::new_with_xavier_rng` and `TwoLayerMLP::new_xavier` for comparative experiments.

4. **Harden Discounted `REINFORCE`, Bellman Solvers, and AI GUI Workbenches**:
   - Buffer $\gamma^t$-weighted policy and critic deltas in pre-allocated `DenseRlWorkspace` buffers without mid-episode mutation (Williams, 1992), bootstrap step-limit truncations with $V(s_{\text{next}})$ (Pardo et al., 2018), remove fabricated `ws.q_table` writes, sample stochastically in `TabularGridPolicy::sample`, bound `EpsilonGreedy` indices, scale symmetric Gauss-Seidel Bellman sweeps for high-$\gamma$ grids, preserve `theta_logits` in `QTableSnapshot`, and reset timeout episodes in `GridWorldTool`.
   - Upgrade `TrainingMonitorTool` with live 2D `PlotImage` class-confidence shading, `MarchingSquares2D` `p = 0.50` decision-boundary contours, `SGD` / `Adam` and `He (ReLU)` / `Xavier / Glorot` controls, and live gradient-norm metrics.

## Consequences

- **Positive**: Restores mathematical fidelity across first-order optimization, cross-entropy gradients, ReLU variance propagation, self-calibration KL losses, discounted policy gradients, and tabular Bellman evaluation without migrating crate paths or allocating inside `VerifiedEngine::run_verified`.
- **Positive**: `TrainingMonitorTool` and `QTableInspectorTool` render live decision boundaries and preserve learned `PolicyGradient` `V_pi` landscapes across interactive `20x20` sculpting with WCAG 2.2 AA (`>= 4.5:1`) contrast across all three themes.
- **Trade-offs**: Public helper additions (`TwoLayerMLP::new_xavier`, `DenseLayer::new_on_stream`, `flush_reinforce_episode_in_place`, `BellmanSolution::recommended_sweeps`, `BellmanSolution::q_pi_at`) expand the public API surface recorded in [`api_baseline.json`](../../apps/unified_verification/api_baseline.json).
