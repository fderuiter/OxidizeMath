# ADR 20261123: Multi-Asset Covariance Kelly, Multivariate QMC Gaussian Copula, Black-Scholes-Merton Greeks, and Portfolio Risk Workbench

## Status
Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), [ADR 20261123](20261123-glicko2-copula-ou-workbenches-and-kelly-net-odds.md), [ADR 20261123](20261123-genz-gaussian-copula-and-parlay-engine.md), and [ADR 20261123](20261123-bidirectional-odds-conversions-and-bet-calculator-sync.md))

## Context
Quantitative portfolio construction, multi-leg Same Game Parlay (SGP) pricing, and derivatives risk analysis in `pure_math::statistics` and `math_explorer_gui` faced four algorithmic and workflow gaps alongside [ADR 20261123](20261123-genz-gaussian-copula-and-parlay-engine.md) and [ADR 20261123](20261123-bidirectional-odds-conversions-and-bet-calculator-sync.md):

1. **Seeded Cranley-Patterson Halton QMC and Equicorrelated Builders for Copula Sweeps**: While [`crates/pure_math/src/pure_math/statistics/copula/core.rs`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs) and [`crates/pure_math/src/pure_math/statistics/copula/gaussian.rs`](../../crates/pure_math/src/pure_math/statistics/copula/gaussian.rs) provide positive-semidefinite Cholesky validation, 20-point Genz bivariate quadrature, and fixed-pair Richtmyer QMC, interactive correlation sweeps and randomized QMC sensitivity checks needed `CorrelationMatrix::equicorrelated` and tunable `OxidizeRng`-seeded Cranley-Patterson Halton QMC (`cdf_with_seed` and `CopulaQmcWorkspace`).
2. **Scalar-Only Kelly Sizing Without Covariance Coupling or Tail Shortfall**: [`crates/pure_math/src/pure_math/statistics/kelly/mod.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/mod.rs) solved single-wager `kelly_fraction` without multi-asset covariance allocation (`Sigma^-1 (mu - r * 1)`), long-only no-short constraints, or `95%`/`99%` Value-at-Risk (`VaR`) and Conditional Value-at-Risk (`CVaR`).
3. **Missing Closed-Form European Option Pricing & Greeks Engine**: `pure_math::statistics` lacked an analytical Black-Scholes-Merton solver for Call and Put valuation alongside first- and second-order Greeks (`Delta`, `Gamma`, `Vega`, `Theta`, `Rho`).
4. **No Unified Multi-Asset Risk & Return-Series Workbench**: `FinancialMathTab` lacked an integrated workbench uniting covariance Kelly frontiers, `2..=12` leg copula tail crashes, `OuAnalyzer` (`estimate_ou_params` + `monte_carlo_forecast`) forecast cones, Black-Scholes Greeks profiles, and `.csv`/`.json` return-series import and risk tear-sheet export.

## Decision
1. **Add Equicorrelated Matrix Construction and Seeded Halton QMC in `pure_math::statistics::copula`**:
   - Add `CorrelationMatrix::equicorrelated` in [`crates/pure_math/src/pure_math/statistics/copula/core.rs`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs) on top of the cached Cholesky positive-semidefinite validator (rejecting indefinite matrices such as `3x3` matrices with `min_eigenvalue = -0.8` via `CopulaError::NotPositiveDefinite`).
   - Add Genz (1992) Cholesky separation-of-variables quasi-Monte Carlo integration using Cranley-Patterson-shifted Halton sequences (`oxidize_core::rng::OxidizeRng`) and antithetic variates in [`crates/pure_math/src/pure_math/statistics/copula/qmc.rs`](../../crates/pure_math/src/pure_math/statistics/copula/qmc.rs) (`cdf_with_seed` delegating `d == 2` to 20-point Genz Gauss-Legendre quadrature, running strictly on the stack with zero heap allocations for `<= 12` legs, and providing `CopulaQmcWorkspace` for `d > 12`).
2. **Extend `pure_math::statistics::kelly` with Covariance Solvers and `VaR`/`CVaR` Calculators**:
   - Add zero-allocation workspace-backed and convenience APIs in [`crates/pure_math/src/pure_math/statistics/kelly/portfolio.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/portfolio.rs) for unconstrained `Sigma^-1 (mu - r * 1)`, diagonal independent Kelly, and projected-gradient long-only Kelly (supporting uncapped `w >= 0` and Duchi-projected leverage-capped `sum w_i <= L_max` constraints).
   - Add empirical in-place, parametric log-normal, and seeded Cholesky Monte Carlo `95%`/`99%` `VaR` and `CVaR` calculators in [`crates/pure_math/src/pure_math/statistics/kelly/tail_risk.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/tail_risk.rs).
3. **Add Closed-Form `pure_math::statistics::black_scholes` Engine and `OptionPricingError` Facade Routing**:
   - Implement `BlackScholesParams`, `OptionType`, `OptionGreeks`, `BlackScholesPair`, `black_scholes_merton`, and `price_and_greeks` in [`crates/pure_math/src/pure_math/statistics/black_scholes.rs`](../../crates/pure_math/src/pure_math/statistics/black_scholes.rs), wiring `OptionPricingError` through `PureMathError` and `math_explorer::error::Error`.
4. **Register `PortfolioCopulaWorkbenchTool` and Verify `BetSizeCalculatorTool::export_datasets`**:
   - Register [`math_explorer_gui/src/tabs/financial_math/portfolio_workbench.rs`](../../math_explorer_gui/src/tabs/financial_math/portfolio_workbench.rs) in `FinancialMathTab` with cached `egui_plot` views for Kelly frontiers, `2..=12` leg copula crash curves, `OuAnalyzer` quantile cones, and Black-Scholes Greeks profiles, plus `web_io::spawn_open_file_dialog` `.csv`/`.json` return import (`<= 2 MiB`, `<= 10,000` rows) and formula-safe `web_io::spawn_save_as_file_dialog` `.csv`/`.json` tear-sheets.
   - Verify direct `BetSizeCalculatorTool::export_datasets` finite-point invariants alongside four-format odds synchronization.

```text
Updated GUI inventory totals:
- InteractiveTools: 88 (67 plot + 8 raster + 13 non-figure; 85 non-shader + 3 shader)
- StarterPresetMetadata cards: 149
```

## Consequences
- **Positive**: Analysts can verify covariance overbetting (`2.7273x` vs `6.0000x` independent leverage on the `3`-asset `rho = 0.60` benchmark), `61.35x` copula tail crash amplification, OU forecast cones, and option Greeks entirely client-side with zero steady-state allocations and compact (`< 2 KiB`) `#tab=...` shareable URLs.
- **Negative / Trade-offs**: Expanding `pure_math` and `math_explorer` public error and solver types requires regenerating `crates/unified_verification/api_baseline.json` and updating explicit GUI inventory counts in the contract test suites.
