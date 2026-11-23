# ADR 0046: Deterministic Genz Gaussian Copula Solvers, Cholesky PSD Validation, and Zero-Allocation Parlay Engine

## Status
Accepted (extends [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), [ADR 0035](0035-kelly-net-odds-seeded-rng-and-paradox-labs.md), and [ADR 0042](0042-hierarchical-umbrella-errors-and-diagnostic-bus-helpers.md))

## Context
[`crates/pure_math/src/pure_math/statistics/copula`](../../crates/pure_math/src/pure_math/statistics/copula/mod.rs) previously exhibited four numerical and architectural limitations for Same Game Parlay (SGP) pricing and stochastic simulation:
1. **Missing Positive-Semidefinite Validation**: [`CorrelationMatrix::new`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs) checked square dimensions, symmetry, unit diagonals, and `[-1, 1]` bounds, but omitted Cholesky or eigenvalue checks—accepting indefinite matrices with negative eigenvalues while leaving `CopulaError::NotPositiveDefinite` unused.
2. **Boundary Failures and Bivariate Midpoint Grid Inaccuracy**: `GaussianCopula::bivariate_normal_cdf` rejected `0.0` and `1.0` marginal probabilities as non-finite z-scores and integrated over a fixed `100x100` rectangular midpoint grid from `-10.0`, producing tail errors and Fréchet-Hoeffding bound violations.
3. **Unimplemented Multi-Leg Evaluation and Hot-Path Heap Allocations**: `GaussianCopula::multivariate_cdf` allocated heap `Vec` buffers and `statrs::MultivariateNormal` before returning an unimplemented error for every parlay with `d != 2`, `sgp_joint_probability` cloned `CorrelationMatrix` on each call, and `#[verified]` error paths formatted heap `String` messages via `format!`.
4. **Missing Deterministic Parlay Simulator and Demo Tail Semantics**: Callers lacked a seeded zero-allocation Cholesky Monte Carlo parlay engine reporting fair decimal odds and SGP multipliers, and [`sports_analytics_demo.rs`](../../math_explorer/examples/sports_analytics_demo.rs) passed `0.99` instead of `0.01` for a 99th-percentile tail event.

## Decision
1. **Eigenvalue-Backed PSD Validation and Cached Lower-Triangular Factors**:
   - Validate non-empty, finite, symmetric, unit-diagonal, and positive-semidefinite correlation matrices in [`CorrelationMatrix::new`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs), returning `Err(CopulaError::NotPositiveDefinite)` whenever `SymmetricEigen` detects an eigenvalue below `-TOLERANCE_HIGH` while accepting rank-deficient comonotonic and countermonotonic boundary matrices (`rho = ±1.0`).
   - Precompute and cache the lower-triangular Cholesky / semidefinite factor inside `CorrelationMatrix` (`cholesky_factor()`) for zero-allocation reuse across pricing and simulation calls.
2. **Exact Boundary Reduction and Zero-Allocation Normal Quantiles**:
   - Short-circuit copula boundary axioms before calling `inverse_standard_normal` so any `0.0` marginal yields `0.0`, `1.0` marginals reduce active dimension, and `d = 1` returns `C(u_1) = u_1`.
   - Back `transforms.rs` with zero-allocation `erfc` / `erfc_inv` helpers and remove `format!` heap allocations from `#[verified]` error paths across `copula/`.
3. **20-Point Genz Bivariate Gauss-Legendre and Richtmyer QMC Multi-Leg Solvers**:
   - Add [`bivariate.rs`](../../crates/pure_math/src/pure_math/statistics/copula/bivariate.rs) implementing Genz (2004) 20-point Gauss-Legendre quadrature split at `|rho| = 0.925` between Plackett angular integration and asymptotic tail integration, clamping outputs to Fréchet-Hoeffding bounds (`< 1e-12` error against `scipy.stats.multivariate_normal`).
   - Add [`multivariate.rs`](../../crates/pure_math/src/pure_math/statistics/copula/multivariate.rs) implementing Genz (1992) separation-of-variables with greedy variable prioritization over an antithetic Baker-tent Richtmyer quasi-Monte Carlo sequence (`< 1e-4` error for 3 to 10 legs).
   - Provide stack-buffered `GaussianCopula::cdf` and clone-free `sgp_joint_probability` up to `MAX_STACK_LEGS` (`16`) alongside `cdf_with_scratch` and `sgp_joint_probability_with_scratch` for arbitrary leg counts with zero heap allocations.
4. **Zero-Allocation Antithetic Cholesky Parlay Engine and Updated Demo**:
   - Add [`parlay.rs`](../../crates/pure_math/src/pure_math/statistics/copula/parlay.rs) exposing `ParlayPricingReport` (`joint_probability`, `standard_error`, `naive_probability`, `correlation_multiplier`, `fair_decimal_odds`), `price_correlated_parlay`, `price_correlated_parlay_with_scratch`, `simulate_correlated_parlay`, and `simulate_correlated_parlay_into` seeded exclusively via `OxidizeRng`.
   - Update [`sports_analytics_demo.rs`](../../math_explorer/examples/sports_analytics_demo.rs) to use `0.01` tail probabilities and demonstrate 3-leg SGP pricing and simulation.

## Consequences
- **Positive**: Indefinite correlation matrices are rejected at construction time with `CopulaError::NotPositiveDefinite`, copula boundary axioms hold exactly, bivariate evaluations achieve `< 1e-12` accuracy, and 3-to-10-leg SGPs achieve `< 1e-4` accuracy.
- **Positive**: `GaussianCopula::cdf_with_scratch`, `sgp_joint_probability`, `price_correlated_parlay`, and `simulate_correlated_parlay_into` execute with zero dynamic heap allocations under `VerifiedAllocator` and `VerifiedEngine::run_verified`, and compile cleanly on `wasm32-unknown-unknown` with zero OS entropy calls.
- **Neutral**: Public additions in `pure_math::statistics::copula` are recorded in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).
