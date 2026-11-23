# ADR 20261123: Promote `#![no_std]` Stochastic Solver Kernel (`SolverWorkspace`) to `math_commons`

## Status

Accepted (extends [ADR 0035](0035-parrondo-paradox-markov-ratchet-lab.md), [ADR 0040](0040-promote-zero-alloc-curve-solvers-to-math-commons.md), and [ADR 20261123](20261123-markov-sequence-and-reliability-workbench.md))

## Context

Discrete-time Markov chains (`MarkovChain`), continuous-time generator chains (`ContinuousMarkovChain`), Parrondo ratchet games (`parrondo`), and Ornstein-Uhlenbeck mean-reversion models (`ou_process` and `CorrelatedRiskScenarioTool`) previously suffered from five numerical and architectural defects:

1. **Unconditional `std` Coupling in `math_commons`**: [`crates/math_commons/Cargo.toml`](../../crates/math_commons/Cargo.toml) unconditionally pulled `diagnostics` and `verified_engine` without a `std` feature gate, preventing Layer 3 foundational solvers from compiling under `#![no_std]` (`core` + `alloc` + `libm`).
2. **Absorbing Uniform Fallback and Periodic Divergence in `MarkovChain::stationary_distribution`**: [`dtmc/mod.rs`](../../crates/pure_math/src/pure_math/statistics/markov/dtmc/mod.rs) returned uniform `1 / n_absorbing` weights whenever absorbing states were present (returning non-unique weights on multi-absorbing chains and ignoring single-sink mixtures) and relied on power iteration $v_{k+1} = P^T v_k$, which oscillates without converging on periodic chains (forcing [`parrondo.rs`](../../crates/pure_math/src/pure_math/statistics/markov/parrondo.rs) to introduce a local `lazy_stationary_distribution` $0.5(I + P)$ damping wrapper). `MarkovChain` also lacked `limiting_distribution` to map an initial distribution through exact transient-to-absorbing probabilities $B = N \cdot R$.
3. **Horizon-Bounded Padé Evaluation and Allocating Jumps in `ContinuousMarkovChain`**: [`ctmc/mod.rs`](../../crates/pure_math/src/pure_math/statistics/markov/ctmc/mod.rs) approximated steady-state vectors by evaluating five allocating Padé(6,6) matrix exponentials at $t \in \{100, 200, 300, 400, 500\}$ with an unscaled `1e-6` residual threshold—failing or returning uniform `0.5` errors on slow reliability rates ($10^{-7}\text{ s}^{-1}$) in `MarkovSequenceWorkbenchTool` and accepting multi-sink chains—while `simulate_trajectory` allocated `weights` and `next_states` `Vec`s on every jump instead of delegating to `GillespieSolver`.
4. **Biased Ornstein-Uhlenbeck Volatility Calibration**: `estimate_ou_params` in [`ou_process/analysis.rs`](../../crates/pure_math/src/pure_math/statistics/ou_process/analysis.rs) returned $\hat{\sigma} = \sqrt{\mathrm{Var}(X)}$ without stationary $\sqrt{2\theta}$ diffusion scaling or exact discrete-time AR(1) conditional residual variance, biasing $\hat{\sigma}$ across slow and fast $\theta$ regimes.
5. **Duplicated Monte Carlo Quantile Loops**: [`correlated_risk/engine.rs`](../../math_explorer_gui/src/tabs/financial_math/correlated_risk/engine.rs) allocated and sorted terminal OU ensembles inline rather than sharing `SolverWorkspace` summary statistics with `OuAnalyzer::monte_carlo_forecast`.

## Decision

1. **Feature-Gate `std` in `crates/math_commons` and Introduce `math_commons::stochastic`**:
   - Gate `diagnostics`, `verified_engine`, and `std` features of `thiserror`, `num-complex`, and `num-traits` behind a default `std` feature in [`crates/math_commons/Cargo.toml`](../../crates/math_commons/Cargo.toml), routing transcendental scalar math through locked `libm` (`0.2.16`) so `cargo check -p math_commons --no-default-features` compiles cleanly under `#![no_std]` + `alloc`.
   - Implement [`crates/math_commons/src/stochastic/`](../../crates/math_commons/src/stochastic/mod.rs) (`mod.rs`, `tarjan.rs`, `qr.rs`, `ou.rs`) around preallocated `SolverWorkspace` buffers (`SccSummary`, `OuCalibration`, `SampleSummary`) re-exported from `math_commons::stochastic`, `pure_math::stochastic`, and `math_explorer::stochastic`.
2. **Iterative Tarjan SCC Sink Isolation + Scale-Normalized Householder QR**:
   - Traverse directed support graphs iteratively inside `SolverWorkspace` (`tarjan.rs`) using scale-invariant row off-diagonal thresholds to isolate closed communicating sink classes without recursion or heap allocation.
   - Return `None` whenever zero or multiple closed sink classes exist (non-unique stationary subspace). When a unique closed class $C$ of size $m$ exists, assemble the column-scaled $(m + 1) \times m$ constrained homogeneous system ($(P_C^T - I)\pi_C = 0$ or $G_C^T \pi_C = 0$ stacked with $\sum_{i \in C} \pi_i = 1$) in `qr.rs`, solve via Householder QR, clamp tiny rounding negatives, renormalize, scatter zeros onto transient states, and verify scale-relative operator residuals below `1e-12`.
3. **Exact AR(1) Ornstein-Uhlenbeck Calibration & Shared Sample Summaries**:
   - Implement `SolverWorkspace::calibrate_ou` and `calibrate_ou` (`ou.rs`) maximizing exact discrete-time conditional Gaussian AR(1) innovations $X_{k+1} - \mu = e^{-\theta \Delta t}(X_k - \mu) + \varepsilon_k$ with $\mathrm{Var}(\varepsilon_k) = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta \Delta t})$ via `-libm::expm1(-2.0 * theta * dt)` and stationary $\sigma = \sqrt{2\theta \cdot \hat{V}_{\mathrm{stat}}}$ scaling across slow, interior, and fast mean-reversion regimes.
   - Implement zero-allocation `SolverWorkspace::summarize_samples` (`ou.rs`) for mean, Bessel-corrected standard deviation, median, `p05`/`p95` order statistics, and exceedance probabilities.
4. **Rewire `pure_math` and `math_explorer_gui` Callers & Delete Workarounds**:
   - Delegate `MarkovChain::stationary_distribution` (`dtmc/stationary.rs`) and `ContinuousMarkovChain::steady_state` (`ctmc/mod.rs`) to `SolverWorkspace`, add `stationary_distribution_with_workspace`, `steady_state_with_workspace`, and `MarkovChain::limiting_distribution` ($B = N \cdot R$), and delete `lazy_stationary_distribution` from [`parrondo.rs`](../../crates/pure_math/src/pure_math/statistics/markov/parrondo.rs).
   - Implement `StochasticSystem<usize>` for `ContinuousMarkovChain<T>` and rewire `ContinuousMarkovChain::simulate_trajectory` onto preallocated `GillespieSolver::with_capacity`.
   - Rewire `estimate_ou_params`, `OuAnalyzer::monte_carlo_forecast`, `summarize_terminal_values_with_workspace`, and [`CorrelatedRiskScenarioTool`](../../math_explorer_gui/src/tabs/financial_math/correlated_risk/engine.rs) onto `SolverWorkspace`.

## Consequences

- **Positive**: Periodic chains (`[0.5, 0.25, 0.25]`) and slow-rate CTMCs (`1e-7` / `3e-7` $\to$ `[0.75, 0.25]`) solve to `< 1e-12` scale-relative residual in a single algebraic pass; multi-absorbing/multi-sink chains rigorously return `None`; OU volatility calibration is unbiased across slow and fast $\theta$ regimes; and `SolverWorkspace` hot paths execute with zero heap allocations under `VerifiedAllocator`.
- **Negative / Trade-offs**: Deleting the `parrondo::lazy_stationary_distribution` workaround and promoting `math_commons::stochastic` + `MarkovChain::limiting_distribution` updates the public API surface, requiring `api_baseline.json` regeneration via `cargo run -p xtask -- regenerate-baseline`.

## Verification

```bash
cargo check -p math_commons --no-default-features
cargo test -p math_commons
cargo test -p pure_math
cargo test -p math_explorer --all-features
cargo test -p math_explorer_gui correlated_risk -- --test-threads=1
cargo test -p math_explorer_gui markov -- --test-threads=1
```
