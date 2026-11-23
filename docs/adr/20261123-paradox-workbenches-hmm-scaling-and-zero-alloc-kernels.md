# ADR 20261123: Standalone Paradox Workbenches, HMM Scaling Likelihood, and Zero-Allocation Statistical Kernels

## Status
Accepted (extends [ADR 0001](0001-dynamic-workspace-discovery.md), [ADR 0035](0035-kelly-net-odds-seeded-rng-and-paradox-labs.md), and [ADR 0038](0038-federated-inverse-challenges-and-hud.md))

## Context
Three verified statistical engines—hierarchical `WinRatioAnalysis` paired with `try_estimate_hazard_ratio` and `TTestIndependent` in `domain_applied`, bivariate `GaussianCopula` / `sgp_joint_probability` in `pure_math::statistics::copula`, and `HiddenMarkovModel` in `pure_math::statistics::markov::hmm`—required dedicated interactive paradox sandboxes, zero-allocation hot-path workspaces, and exact likelihood scaling:
1. `HiddenMarkovModel::forward` previously discarded Rabiner's per-step scaling factors $c_t = P(Y_t \mid Y_{0:t-1})$ and summed the already normalized final column $\hat{\alpha}_{T-1}$, erroneously returning `1.0` for every valid sequence instead of $P(Y_{0:T-1}) = \prod_t c_t$.
2. `GaussianCopula::integrate_bivariate_normal` integrated over a truncated $100 \times 100$ midpoint grid on $[-10, z]^2$, losing precision in extreme quantiles such as $u = 0.999$ and lacking a heavy-tailed `StudentTCopula` counterpart.
3. Repeated per-frame 2D regime-map sweeps and trellis branches require preallocated flat buffers so hot recalculation loops execute with zero dynamic heap allocations under `VerifiedAllocator`.

## Decision
1. **Exact Scaled Forward Likelihood in `HiddenMarkovModel`**:
   - Update [`forward_backward.rs`](../../crates/pure_math/src/pure_math/statistics/markov/hmm/forward_backward.rs) so `HiddenMarkovModel::forward` returns $\prod_{t=0}^{T-1} c_t$, add `HiddenMarkovModel::forward_log_likelihood` returning $\sum_t \ln c_t$, and guard zero scaling factors in `backward_probabilities`.
2. **Zero-Allocation Scratch Kernels & 16-Point Gauss-Legendre Copulas**:
   - Introduce [`HmmScratch`](../../crates/pure_math/src/pure_math/statistics/markov/hmm/workspace.rs) for in-place forward, backward, smoothing, filtering, marginal MAP, and Viterbi evaluation.
   - Introduce [`student_t.rs`](../../crates/pure_math/src/pure_math/statistics/copula/student_t.rs) implementing 16-point Gauss-Legendre Plackett quadrature for bivariate Gaussian CDFs, Genz (2004) trigonometric quadrature from the Fréchet-Hoeffding bounds $\rho = \pm 1$ for `StudentTCopula`, and `BivariateCopulaScratch` for 2D $(\rho, u)$ tail-amplification grids.
   - Introduce [`WinRatioScratch`](../../crates/domain_applied/src/applied/win_ratio/scratch.rs) and flat-stride `WinRatioAnalysis` evaluators eliminating per-patient `Vec<f64>` allocations.
3. **Three Standalone `InteractiveTool` Paradox Workbenches & Inverse Challenges**:
   - Mount `WinRatioParadoxLab` in `ClinicalTrialsTab` (`win_ratio_severity_reversal`), `CopulaTailRiskLab` in `FinancialMathTab` (`copula_heavy_tail_kelly`), and `HiddenMarkovParadoxTool` in `FinancialMathTab` (`hmm_trellis_branch_rescue`), preserving existing `SurvivalAnalysisTool` and `BetSizeCalculatorTool` URL/preset schemas unchanged.

## Consequences
- **Positive**: Reproduces exact audit benchmarks (`HR = 3.8623` vs `WR = 1.6667`, `0.84%` to `11.29%` SGP swing and `5.83x` Student-t tail crash at `u = 0.999`, and `forward = 0.0693765` with forbidden `[0, 1, 1]` MAP vs `[0, 2, 1]` Viterbi) with zero heap allocations inside `VerifiedEngine::run_verified`.
- **Neutral**: Expands the federated GUI registry to 79 `InteractiveTool` entries, 112 `StarterPresetMetadata` cards, 57 plot-exportable tools, and 8 playable `ChallengeMetadata` inverse challenges.
