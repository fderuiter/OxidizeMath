# ADR 20261123: Multi-Asset Covariance Kelly, Multivariate QMC Gaussian Copula, Black-Scholes-Merton Greeks, and Portfolio Risk Workbench

## Status
Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), and [ADR 20261123](20261123-glicko2-copula-ou-workbenches-and-kelly-net-odds.md))

## Context
Quantitative portfolio construction, multi-leg Same Game Parlay (SGP) pricing, and derivatives risk analysis in `pure_math::statistics` and `math_explorer_gui` faced four algorithmic and workflow limitations:

1. **Unvalidated Indefinite Correlation Matrices and Stubbed Multivariate Copulas**: [`crates/pure_math/src/pure_math/statistics/copula/core.rs`](../../crates/pure_math/src/pure_math/statistics/copula/core.rs) checked symmetry and unit diagonal entries in `CorrelationMatrix::new` without verifying positive-definiteness, permitting indefinite matrices (such as a `3x3` matrix with `min_eigenvalue = -0.8`), while [`crates/pure_math/src/pure_math/statistics/copula/gaussian.rs`](../../crates/pure_math/src/pure_math/statistics/copula/gaussian.rs) returned a `NumericalError` stub for `d >= 3` legs and eagerly allocated a `String` inside `bivariate_cdf`.
2. **Scalar-Only Kelly Sizing Without Covariance Coupling or Tail Shortfall**: [`crates/pure_math/src/pure_math/statistics/kelly/mod.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/mod.rs) solved single-wager `kelly_fraction` without multi-asset covariance allocation (`Sigma^-1 (mu - r * 1)`), long-only no-short constraints, or `95%`/`99%` Value-at-Risk (`VaR`) and Conditional Value-at-Risk (`CVaR`).
3. **Missing Closed-Form European Option Pricing & Greeks Engine**: `pure_math::statistics` lacked an analytical Black-Scholes-Merton solver for Call and Put valuation alongside first- and second-order Greeks (`Delta`, `Gamma`, `Vega`, `Theta`, `Rho`).
4. **No Unified Multi-Asset Risk & Return-Series Workbench**: `FinancialMathTab` lacked an integrated workbench uniting covariance Kelly frontiers, `3..=12` leg copula tail crashes, `OuAnalyzer` (`estimate_ou_params` + `monte_carlo_forecast`) forecast cones, Black-Scholes Greeks profiles, and `.csv`/`.json` return-series import and risk tear-sheet export, while `BetSizeCalculatorTool` relied on thread-local post-render `egui_plot` buffers instead of overriding `InteractiveTool::export_datasets` directly.

## Decision
1. **Enforce Cholesky Positive-Definiteness and Zero-Allocation QMC Integration in `pure_math::statistics::copula`**:
   - Validate strict positive-definiteness via Cholesky factorization in `CorrelationMatrix::new` (returning `CopulaError::NotPositiveDefinite` on zero or negative pivots) and cache the lower-triangular factor `L`.
   - Implement Genz (1992) Cholesky separation-of-variables quasi-Monte Carlo integration using Cranley-Patterson-shifted Halton sequences (`oxidize_core::rng::OxidizeRng`) and antithetic variates in [`crates/pure_math/src/pure_math/statistics/copula/qmc.rs`](../../crates/pure_math/src/pure_math/statistics/copula/qmc.rs), running strictly on the stack with zero heap allocations for `<= 12` legs (`CopulaQmcWorkspace` for `d > 12`) and eliminating the eager `String` allocation in `bivariate_cdf`.
2. **Extend `pure_math::statistics::kelly` with Covariance Solvers and `VaR`/`CVaR` Calculators**:
   - Add zero-allocation workspace-backed and convenience APIs in [`crates/pure_math/src/pure_math/statistics/kelly/portfolio.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/portfolio.rs) for unconstrained `Sigma^-1 (mu - r * 1)`, diagonal independent Kelly, and projected-gradient long-only Kelly (supporting uncapped `w >= 0` and Duchi-projected leverage-capped `sum w_i <= L_max` constraints).
   - Add empirical in-place, parametric log-normal, and seeded Cholesky Monte Carlo `95%`/`99%` `VaR` and `CVaR` calculators in [`crates/pure_math/src/pure_math/statistics/kelly/tail_risk.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/tail_risk.rs).
3. **Add Closed-Form `pure_math::statistics::black_scholes` Engine and `OptionPricingError` Facade Routing**:
   - Implement `BlackScholesParams`, `OptionType`, `OptionGreeks`, `BlackScholesPair`, `black_scholes_merton`, and `price_and_greeks` in [`crates/pure_math/src/pure_math/statistics/black_scholes.rs`](../../crates/pure_math/src/pure_math/statistics/black_scholes.rs), wiring `OptionPricingError` through `PureMathError` and `math_explorer::error::Error`.
4. **Register `PortfolioCopulaWorkbenchTool` and Direct `BetSizeCalculatorTool::export_datasets`**:
   - Register [`math_explorer_gui/src/tabs/financial_math/portfolio_workbench.rs`](../../math_explorer_gui/src/tabs/financial_math/portfolio_workbench.rs) in `FinancialMathTab` with cached `egui_plot` views for Kelly frontiers, `2..=12` leg copula crash curves, `OuAnalyzer` quantile cones, and Black-Scholes Greeks profiles, plus `web_io::spawn_open_file_dialog` `.csv`/`.json` return import (`<= 2 MiB`, `<= 10,000` rows) and formula-safe `web_io::spawn_save_as_file_dialog` `.csv`/`.json` tear-sheets.
   - Implement `InteractiveTool::export_datasets` directly on `BetSizeCalculatorTool` so plot dataset exports succeed even prior to rendering a frame.

```text
Updated GUI inventory totals:
- InteractiveTools: 77 (55 plot + 7 raster + 15 non-figure; 74 non-shader + 3 shader)
- StarterPresetMetadata cards: 110
```

## Consequences
- **Positive**: Analysts can verify covariance overbetting (`2.7273x` vs `6.0000x` independent leverage on the `3`-asset `rho = 0.60` benchmark), `61.35x` copula tail crash amplification, OU forecast cones, and option Greeks entirely client-side with zero steady-state allocations and compact (`< 2 KiB`) `#tab=...` shareable URLs.
- **Negative / Trade-offs**: Expanding `pure_math` and `math_explorer` public error and solver types requires regenerating `crates/unified_verification/api_baseline.json` and updating explicit GUI inventory counts in the contract test suites.
