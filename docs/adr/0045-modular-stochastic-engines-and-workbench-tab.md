# ADR 0045: Modular Stochastic Engines and Dedicated Workbench Tab

## Status

Accepted (2026-11-23)

## Context

Previously, discrete- and continuous-time stochastic models in [`crates/pure_math/src/pure_math/statistics/markov/`](../../crates/pure_math/src/pure_math/statistics/markov/) and [`crates/pure_math/src/pure_math/statistics/ou_process/`](../../crates/pure_math/src/pure_math/statistics/ou_process/) had numerical accuracy and modularity gaps, and lacked interactive exploration outside the narrow Parrondo's Paradox tools:

1. `MarkovChain::stationary_distribution` used power iteration on $P^T$ (oscillating on periodic chains) and fabricated a uniform $1 / n_{\text{absorbing}}$ vector on multi-absorbing chains instead of distinguishing unique stationary distributions from initial-state-dependent limiting absorption distributions $B = (I - Q)^{-1} R$.
2. `ContinuousMarkovChain::steady_state` iterated matrix exponentials $\exp(G \cdot 2^k)$ up to $t = 2^{30}$, failing to mix on slow generators ($10^{-4}$ to $10^{-7}$ rates) while performing expensive Padé scaling-and-squaring loops.
3. `HiddenMarkovModel::forward` returned the terminal scaled $\alpha_T$ sum ($\approx 1.0$) instead of likelihood, `viterbi` multiplied raw probabilities (underflowing on sequences of $500+$ steps), and parameter estimation via Baum-Welch EM was absent.
4. `estimate_ou_params` divided residual SSE by $\sqrt{\Delta t}$ instead of scaling innovation variance by $\frac{2\hat{\theta}}{1 - \hat{\rho}_1^2}$, biasing $\hat{\sigma}$ when $\theta \Delta t$ was non-negligible, and lacked an exact Gaussian transition solver (`ExactOuSolver`).
5. Poisson arrival processes (`HomogeneousPoisson`, `InhomogeneousPoisson`) and Markovian $M/M/1$ and $M/M/c$ Erlang-C birth-death queues were absent.

## Decision

1. **Modular DTMC & CTMC Nullspace Solvers and Zero-Allocation Trajectory Stepping:**
   - Split [`dtmc/mod.rs`](../../crates/pure_math/src/pure_math/statistics/markov/dtmc/mod.rs) into focused submodules (`scc.rs`, `period.rs`, `hitting.rs`, `stationary.rs`, `simulation.rs`). Iterative Tarjan strongly connected components (`communicating_classes`) identify closed recurrent classes; `stationary_distribution` solves $(P^T - I)\pi = 0, \sum_i \pi_i = 1$ via rank-checked QR decomposition only when a unique closed class exists (`None` otherwise), while `limiting_distribution` projects initial weights through $B = (I - Q)^{-1} R$.
   - Added [`ctmc/steady_state.rs`](../../crates/pure_math/src/pure_math/statistics/markov/ctmc/steady_state.rs) and [`ctmc/simulation.rs`](../../crates/pure_math/src/pure_math/statistics/markov/ctmc/simulation.rs) solving rate-preconditioned $G^T \pi = 0, \sum_i \pi_i = 1$ via QR decomposition and simulating zero-allocation Gillespie trajectories (`simulate_gillespie_into`) with `OxidizeRng`.

2. **Poisson Thinning, $M/M/c$ Queueing, Log-Space HMMs, and Exact OU Calibration:**
   - Added [`poisson.rs`](../../crates/pure_math/src/pure_math/statistics/poisson.rs) (`HomogeneousPoisson`, `InhomogeneousPoisson` with fixed `IntensityProfile` variants via zero-allocation Lewis-Shedler thinning) and [`queueing.rs`](../../crates/pure_math/src/pure_math/statistics/queueing.rs) (`Mm1Queue`, `MmcQueue` overflow-safe Erlang-C recurrences, `to_truncated_ctmc`, and zero-allocation birth-death stepping).
   - Updated [`hmm/`](../../crates/pure_math/src/pure_math/statistics/markov/hmm/) with `forward_log_likelihood` ($\sum_t \ln c_t$), `forward = exp(forward_log_likelihood)`, log-space `viterbi` with $-\infty$ zero guards, and simplex-floored `baum_welch` EM ([`baum_welch.rs`](../../crates/pure_math/src/pure_math/statistics/markov/hmm/baum_welch.rs)).
   - Added [`exact_solver.rs`](../../crates/pure_math/src/pure_math/statistics/ou_process/exact_solver.rs) (`ExactOuSolver`) and updated `estimate_ou_params` in [`analysis.rs`](../../crates/pure_math/src/pure_math/statistics/ou_process/analysis.rs) with exact discrete AR(1) innovation variance scaling and stationary $\sqrt{2\hat{\theta}\operatorname{Var}(X)}$ fallback.

3. **Dedicated `StochasticProcessesTab` in `math_explorer_gui`:**
   - Added [`math_explorer_gui/src/tabs/stochastic_processes/mod.rs`](../../math_explorer_gui/src/tabs/stochastic_processes/mod.rs) (`// @explorer_feature = "pure_math"`, `#tab=stochastic_processes`) hosting `DtmcTensorWorkbenchTool`, `CtmcQueueWorkbenchTool`, and `HmmOuWorkbenchTool` with six starter presets, `ToolStateBounds` URL hydration, AccessKit canvas landmarks, and `ExportCapabilities::plot()` exports.

## Consequences

- Periodic DTMCs and slow-rate ($10^{-7}$) CTMCs converge to exact machine-precision stationary distributions ($\le 10^{-12}$) in a single QR solve, and multi-absorbing chains cleanly separate `stationary_distribution() == None` from `limiting_distribution`.
- All discrete-event stepping kernels (`step_oxidize`, `simulate_into`, `simulate_gillespie_into`, `simulate_arrivals_into`, `simulate_trajectory_into`, `ExactOuSolver::simulate_into`) execute with zero heap allocations under `VerifiedAllocator`.
- Users can inspect DTMC SCC classes, `TransitionTensor` shot-clock interpolation, CTMC/$M/M/c$/Poisson dynamics, 600-step HMM Viterbi/Baum-Welch inference, and exact OU calibration on desktop and `wasm32-unknown-unknown`.
