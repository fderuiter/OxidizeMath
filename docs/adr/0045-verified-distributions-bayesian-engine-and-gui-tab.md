# ADR 0045: Verified Probability Distributions, Conjugate & Zero-Allocation Bayesian Engine, and Probability & Statistics Tab

## Status

Accepted (2026-11-23)

## Context

Previously, `pure_math::statistics` exposed isolated point-purpose statistical routines without a unified probability distribution trait hierarchy, inverse-CDF quantile solvers, or Maximum Likelihood / Method-of-Moments estimators:

- [`copula`](../../crates/pure_math/src/pure_math/statistics/copula/mod.rs) accepted symmetric unit-diagonal matrices without checking positive-definiteness, approximated bivariate Gaussian copulas on a truncated `100 x 100` midpoint grid over `[-10, z]` (losing accuracy in deep tails), and left `multivariate_cdf` unimplemented for dimensions `d >= 3`.
- [`zip_regression`](../../crates/pure_math/src/pure_math/statistics/zip_regression/mod.rs) clamped `simple_zip_fit` zero-inflation estimates at `0.95` and lacked an Expectation-Maximization (EM) covariate optimizer on `ZipRegression`.
- `domain_applied` lacked normalized multi-label Bayesian posteriors in FreeSurfer segmentation (`papers/freesurfer.tex`) and Bayesian two-arm clinical trial comparisons alongside frequentist hypothesis tests (`papers/clinical_trials_statistics.tex`), while `math_explorer_gui` had no interactive Probability & Statistics workbench.

## Decision

1. **Stack-Allocated `ContinuousDistribution` & `DiscreteDistribution` Suite (`pure_math::statistics::distributions`):**
   - Added [`distributions/mod.rs`](../../crates/pure_math/src/pure_math/statistics/distributions/mod.rs) defining `ContinuousDistribution` and `DiscreteDistribution` across 14 parametric families (`NormalDist`, `BetaDist`, `GammaDist`, `StudentTDist`, `ChiSquaredDist`, `ExponentialDist`, `UniformDist`, `LogNormalDist`, `WeibullDist`, `BernoulliDist`, `BinomialDist`, `PoissonDist`, `GeometricDist`, `NegativeBinomialDist`) plus `ZipDistribution`.
   - Every distribution provides exact `pdf`/`pmf`, `ln_pdf`/`ln_pmf`, `cdf`, safeguarded Acklam / Newton-bisection `quantile`, analytical moments, deterministic `OxidizeRng` sampling with zero heap allocations, and `fit_mle` / `fit_mom` parameter estimators backed by stack-only [`DistributionError`](../../crates/pure_math/src/error/statistics_errors.rs).

2. **Cholesky-Validated Copulas, Genz QMC, and Damped-IRLS ZIP EM (`copula`, `zip_regression`):**
   - Enforced strict Cholesky factorization (`CopulaError::NotPositiveDefinite`) in [`CorrelationMatrix::new`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs), replaced grid integration with 16-point Gauss-Legendre trigonometric Plackett quadrature for `d = 2`, and implemented zero-allocation Genz (1992) Cholesky separation-of-variables Quasi-Monte Carlo (`cdf` / `cdf_with_rng`) respecting Fréchet-Hoeffding bounds across dimensions `2..=16`.
   - Removed the artificial `0.95` zero-inflation cap in `simple_zip_fit`, added $O(1)$ `cdf`, `ln_pmf`, `quantile`, and seeded `sample` on `ZipDistribution`, and implemented damped-IRLS Expectation-Maximization in [`zip_regression/em_solver.rs`](../../crates/pure_math/src/pure_math/statistics/zip_regression/em_solver.rs) returning `ZipFitResult` or `ZipError::ConvergenceFailed`.

3. **Conjugate Bayesian Models, Zero-Allocation MCMC, Applied Bridges, and GUI Tab:**
   - Added [`bayesian/mod.rs`](../../crates/pure_math/src/pure_math/statistics/bayesian/mod.rs) implementing `BetaBinomial`, `GammaPoisson`, `NormalNormal`, `NormalInverseGamma`, and `DirichletMultinomial<const K: usize>`, equal-tailed and golden-section HPD `CredibleInterval` solvers, Jeffreys `BayesFactor` classification, and pre-allocated zero-allocation `MetropolisHastings1D` / `MetropolisHastingsND` samplers.
   - Added `normalized_bayesian_posteriors` and `bayesian_classify_labels` in [`freesurfer/segmentation.rs`](../../crates/domain_applied/src/applied/freesurfer/segmentation.rs) and `bayesian_two_arm_proportions` / `bayesian_two_arm_continuous` in [`clinical_trials/bayesian_testing.rs`](../../crates/domain_applied/src/applied/clinical_trials/bayesian_testing.rs), re-exporting all new modules and error variants through [`math_explorer`](../../math_explorer/src/error.rs).
   - Added auto-discovered [`ProbabilityStatisticsTab`](../../math_explorer_gui/src/tabs/probability_statistics/mod.rs) (`#tab=probability_statistics`) hosting `DistributionExplorerTool`, `BayesianInferenceTool`, `CopulaDependenceTool`, and six federated `StarterPresetMetadata` cards.

## Consequences

- Distribution evaluation, quantile inversion, copula Genz QMC integration, FreeSurfer multi-label classification, and Metropolis-Hastings MCMC stepping run with zero dynamic heap allocations under `VerifiedAllocator`.
- Non-positive-definite correlation matrices and non-convergent ZIP EM fits fail explicitly with typed diagnostics (`CopulaError::NotPositiveDefinite`, `ZipError::ConvergenceFailed`).
- Desktop and `wasm32` users gain interactive parametric distribution fitting, conjugate-vs-MCMC posterior verification, and 2D–5D copula dependence exploration with full URL sharing and deferred PNG/SVG figure exports.
