# ADR 0045: Preallocated Monte Carlo, Randomized QMC, and Nonparametric Bootstrap Suite

## Status
Accepted (2026-11-23)

## Context

Downstream stochastic simulators across `pure_math`, `domain_applied`, and `math_explorer_gui` (`OuAnalyzer::monte_carlo_forecast`, `AuctionSimulator::simulate_optimal_revenue_with_rng`, `ParrondoEnsemble`, `GaussianCopula`, `kaplan_meier`, and `BankrollEnsembleLab`) lacked a unified, zero-allocation Monte Carlo variance-reduction, quasi-Monte Carlo (QMC), and nonparametric bootstrap foundation:

1. **Zero-allocation & `oxidize_core` immutability constraints:** `crates/oxidize_core` is frozen, and hot simulation loops must execute with `0` heap allocations under `VerifiedAllocator` while seeding every random draw, shift, and bootstrap resample deterministically through `OxidizeRng`.
2. **Numerical edge-case safety & uncertainty quantification:** `OuAnalyzer::monte_carlo_forecast` previously lacked `n_paths == 0` / `NaN` guards and standard-error confidence intervals; `simulate_optimal_revenue_with_rng` divided by `0` when `n_simulations == 0`; `Trapezoidal` integration returned `0.0` error estimates; and `GaussianCopula` was limited to 2D bivariate grids.
3. **Asymmetric finite-sample inference:** Skewed option payoffs, wealth distributions, and censored clinical survival curves exhibit strong skewness where symmetric normal or naive percentile intervals undercover relative to Efron's bias-corrected and accelerated (`BCa`) bootstrap.

## Decision

1. **`pure_math::statistics::qmc` (`HaltonSequence` & Joe-Kuo `SobolSequence`):**
   - Implement prime-base radical-inverse `HaltonSequence` up to 64 dimensions and Gray-code `O(d)` `SobolSequence` up to 32 dimensions backed by embedded `new-joe-kuo-6.21201` direction numbers (`sobol_data.rs`), starting at Gray-code index `0` with half-ULP midpoint offset so $2^m$ prefixes form exact $(t, m, s)$-digital nets strictly inside `(0, 1)^d`, plus `OxidizeRng`-seeded Cranley-Patterson torus shifts.

2. **`pure_math::statistics::monte_carlo` (`MonteCarloWorkspace` & Kong `ESS`):**
   - Provide `MonteCarloEstimate`, `MonteCarloError` (`Copy`, zero-allocation), `effective_sample_size` ($(\sum w_i)^2 / \sum w_i^2$ with `DegenerateWeights` guard), and preallocated `MonteCarloWorkspace` supporting crude IID, antithetic variates, optimal-$\hat{\beta}^*$ control variates, self-normalized importance sampling (delta-method SE + Kong `ESS`), Latin Hypercube sampling, and randomized Halton/Sobol QMC replicates.

3. **Hyperrectangle `MultivariateIntegrator` & Genz `GaussianCopula::multivariate_cdf`:**
   - Extend `pure_math::analysis::integration` with Richardson $O(h^2)$ error estimates on 1D `Trapezoidal` quadrature and preallocated `MultivariateIntegrator` over $[a_1, b_1] \times \dots \times [a_d, b_d]$.
   - Implement `GaussianCopula::multivariate_cdf` via SPD Cholesky factorization and Genz (1992) greedy variable ordering and separation-of-variables over Cranley-Patterson shifted `SobolSequence`.

4. **`pure_math::statistics::bootstrap` & Domain Hardening:**
   - Implement `BootstrapWorkspace` computing nonparametric `Percentile` and jackknife-accelerated `BCa` intervals (`z0_hat`, `accel_hat`) with zero heap allocations during hot runs (`*_with_scratch`).
   - Harden `OuAnalyzer::monte_carlo_forecast` (`total_cmp`, `n_paths == 0` / `NaN` guards, SE & 95% CI fields), `AuctionSimulator` (`n_simulations == 0` / `NaN` guards + `simulate_optimal_revenue_stats`), `ParrondoEnsemble` (`stderr_trajectory`, `terminal_estimate`), and `kaplan_meier_bootstrap` pointwise survival bands in `domain_applied::applied::clinical_trials::survival_analysis`.
   - Re-export `bootstrap`, `monte_carlo`, and `qmc` cleanly through `math_explorer`.

5. **`MonteCarloWorkbenchTool` & `BankrollEnsembleLab` in `FinancialMathTab`:**
   - Add `MonteCarloWorkbenchTool` (`id: "MonteCarloWorkbenchTool"`, `display_name: "Monte Carlo & Bootstrap Workbench"`) with two `StarterPresetMetadata` cards, `MAX_ITERATIONS` (`5000`) bounds, live Kong `ESS` diagnostics, log-log RMSE convergence plots, and `Percentile` vs `BCa` replicate histograms, plus zero-allocation terminal median 95% `BCa` CIs in `BankrollEnsembleLab`.

## Verification Commands

```bash
cargo test -p pure_math --lib -- statistics:: analysis::integration::
cargo test -p domain_applied --lib -- mechanism_design survival_analysis
cargo test -p math_explorer --all-features
cargo test -p math_explorer_gui monte_carlo_workbench
cargo run -p xtask -- regenerate-baseline
cargo run -p xtask -- lint-all
```

## Consequences

- **Positive:** All Monte Carlo, randomized QMC, Genz copula, and `BCa` bootstrap hot paths execute with strictly `0` heap allocations under `VerifiedAllocator` while leaving `crates/oxidize_core` untouched.
- **Positive:** Smooth multivariate integrals achieve $O(N^{-1} (\log N)^d)$ randomized Sobol convergence, skewed estimators gain second-order accurate `BCa` intervals, and `OuAnalyzer` / `AuctionSimulator` edge cases are guarded against `NaN` and zero-sample panics.
- **Neutral:** Registered `InteractiveTool`, `StarterPresetMetadata`, and plot-export tool counts in `math_explorer_gui` contract suites advance from `72 -> 73`, `103 -> 105`, and `49 -> 50`, respectively.
