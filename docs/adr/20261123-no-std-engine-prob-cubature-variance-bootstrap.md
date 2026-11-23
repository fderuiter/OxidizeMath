# ADR 20261123: `#![no_std]` Probabilistic Engine for QMC Cubature, Variance Reduction, and Bootstrap Inference

## Status

Accepted (2026-11-23)

## Context

Probabilistic numerical estimators across `pure_math`, `domain_applied`, and `math_explorer_gui` previously duplicated quasi-Monte Carlo (QMC) lattice integration, antithetic Cholesky sampling, and ad-hoc resampling logic while relying on `std` heap allocations:

1. **Fragmented QMC & Copula Sampling**: Richtmyer irrational lattice Genz cubature lived inside `crates/pure_math/src/pure_math/statistics/copula/multivariate.rs`, while correlated Bernoulli antithetic sampling lived inside `copula/parlay.rs` without reusable control-variate, importance-sampling, Halton, or Sobol primitives.
2. **Embedded & WebAssembly Portability**: Core stochastic sequences in `crates/oxidize_core` unconditionally pulled in `std` filesystem and logging modules, preventing `#![no_std]` downstream engines from sharing deterministic SplitMix64/xoshiro256++ and low-discrepancy generators over caller-owned scratch slices.
3. **Ornstein-Uhlenbeck Forecast Instabilities & Allocations**: `OuAnalyzer::monte_carlo_forecast` in `crates/pure_math/src/pure_math/statistics/ou_process/analysis.rs` allocated `n_paths` full trajectory `Vec<f64>` buffers only to inspect terminal states, sorted floating-point slices with `partial_cmp().unwrap()`, and panicked on `n_paths == 0` or `NaN` inputs.
4. **Missing Nonparametric Uncertainty Quantification**: Kaplan-Meier survival curves (`domain_applied::applied::clinical_trials::survival_analysis`), censored win-ratio power sweeps (`power_sweep`), Parrondo ratchets (`markov::parrondo::ensemble`), mechanism-design auction estimators (`game_theory::mechanism_design`), and the GUI bankroll volatility-drag lab (`BankrollEnsembleLab`) lacked zero-allocation percentile and Efron bias-corrected and accelerated (BCa) bootstrap intervals and variance-reduced standard errors.

## Decision

1. **Gate `std` in `oxidize_core` and Add `#![no_std]` Low-Discrepancy Sequences**:
   - Add `#![cfg_attr(not(feature = "std"), no_std)]` to [`crates/oxidize_core/src/lib.rs`](../../crates/oxidize_core/src/lib.rs) with `default = ["std"]`, gating OS-dependent modules (`fs`, `logging`, `mesh`, `traceability`) behind `#[cfg(feature = "std")]`.
   - Expose `#![no_std]` Richtmyer Weyl/Kronecker irrational generators (with Baker tent transformation and Cranley-Patterson modulo-1 shift), prime-base radical-inverse `HaltonSequence`, and Gray-code Joe-Kuo `SobolSequence` (up to 40 dimensions with digital XOR shift) alongside `OxidizeRng` in [`crates/oxidize_core/src/rng.rs`](../../crates/oxidize_core/src/rng.rs).
2. **Introduce `#![no_std]` `crates/02_engines/engine_prob`**:
   - Create [`crates/02_engines/engine_prob`](../../crates/02_engines/engine_prob/README.md) with `#![cfg_attr(not(feature = "std"), no_std)]`, optional `alloc` (`OwnedSolverWorkspace`) and `std` (`std::error::Error`) features, `libm` transcendental math, zero OS entropy, and zero heap allocations in hot loops.
   - Define `SolverWorkspace<'a>` over borrowed `&'a mut [f64]` and `&'a mut [usize]` scratch buffers with non-allocating `split_floats`, `split_indices`, and `reborrow` cursors.
   - Implement multidimensional QMC cubature (`integrate_unit_cube`, `integrate_hyperrect`, `mvn_genz_richtmyer_cdf`, `gaussian_copula_richtmyer_cdf`, `genz_semidefinite_richtmyer_qmc`) with Cranley-Patterson and digital shift error bounds (`CubatureEstimate`).
   - Implement variance-reduction kernels (`estimate_antithetic_uniform`, `estimate_antithetic_normal`, `simulate_correlated_bernoulli_antithetic`, `simulate_orthant_antithetic_col_major`, `control_variate_from_slices`, `estimate_control_variate`, `self_normalized_is_from_slices`, `estimate_self_normalized_is`).
   - Implement zero-allocation nonparametric `bootstrap_indices`, `percentile_bootstrap`, and Efron `bca_bootstrap` (with leave-one-out jackknife acceleration $\hat{a}$ and bias-correction $z_0$) plus mean and median shortcuts.
3. **Repair `OuAnalyzer` Forecasting & Unify Downstream Consumers**:
   - Add zero-allocation terminal steppers (`simulate_terminal`, `simulate_terminals_into`, `simulate_antithetic_terminal_pair`, `discrete_terminal_mean`) to `OuSolver` and refactor `OuAnalyzer::monte_carlo_forecast` to guard `n_paths == 0` and non-finite inputs, sort via `f64::total_cmp`, report standard errors, and expose checked `try_monte_carlo_forecast` and antithetic + control-variate `try_monte_carlo_forecast_variance_reduced`.
   - Expose `#[verified]` multidimensional cubature in `pure_math::analysis::cubature` (re-exported in `analysis::integration`), delegate `GaussianCopula::cdf` and `simulate_correlated_parlay_into` to `engine_prob`, and re-export `engine_prob` as `math_explorer::prob` alongside `ProbError` in the `pure_math` and `math_explorer` umbrella errors.
   - Adopt `engine_prob` across `ParrondoEnsemble`, `ruin::ensemble` (extracting `wilson.rs` to keep module files concise), `mechanism_design`, `win_ratio::power_sweep`, `clinical_trials::survival_analysis` (`kaplan_meier_bootstrap_bands`), `BankrollEnsembleLab` in `math_explorer_gui`, and `papers/clinical_trials_statistics.tex`.

## Consequences

- **Positive**:
  - `engine_prob` and `oxidize_core` compile cleanly with `--no-default-features` on bare-metal and `wasm32-unknown-unknown` targets with zero heap allocations inside QMC, variance-reduction, and bootstrap loops.
  - `OuAnalyzer::monte_carlo_forecast` is total over edge-case inputs (`n_paths == 0`, `NaN`, `±∞`), avoids `O(n_paths * steps)` heap allocations, and achieves substantial variance reduction via paired antithetic shocks and exact discrete-step control variates.
  - Clinical survival curves, win-ratio power sweeps, auction revenue estimators, Parrondo ratchets, and interactive Kelly bankroll ensembles gain unified, reproducible uncertainty quantification.
- **Trade-offs**:
  - Callers invoking `#![no_std]` `engine_prob` kernels directly size scratch slices via `workspace_floats_for_*` and `workspace_indices_for_*` const helpers (or use `OwnedSolverWorkspace` when `alloc` is enabled), while high-level `pure_math` and `domain_applied` facades manage scratch workspaces behind `#[verified]` boundaries.
