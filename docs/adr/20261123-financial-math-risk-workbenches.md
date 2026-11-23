# ADR 20261123: Financial Math Risk Workbenches, Zero-Allocation Stochastic Helpers, and Fractional Kelly Growth Diagnostics

## Status
Accepted (extends [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), [ADR 20261123](20261123-bidirectional-odds-conversions-and-bet-calculator-sync.md), [ADR 20261123](20261123-genz-gaussian-copula-and-parlay-engine.md), and [ADR 20261123](20261123-markov-sequence-and-reliability-workbench.md))

## Context
While `pure_math::statistics` already shipped Cholesky-validated Genz Gaussian copulas (`copula`), Ornstein-Uhlenbeck MLE + Euler-Maruyama solvers (`ou_process`), Continuous-Time Markov Chain generators + Hidden Markov Model Baum-Welch / Viterbi algorithms (`markov`), and net-odds Kelly sizing (`kelly`), `FinancialMathTab` lacked interactive quantitative risk workbenches for:
1. Multivariate correlated parlay / same-game-parlay (`2..=8` legs) tail-risk pricing, bivariate iso-CDF contours, and zero-allocation correlated uniform simulation (`CopulaRiskTool`).
2. Empirical UTF-8 / UTF-8 BOM CSV time-series calibration (`estimate_ou_params`), zero-allocation `EulerMaruyama::simulate_into` Monte Carlo confidence bands, URL share-size diagnostics, and calibrated Rust/Python/LaTeX snippet export (`OuProcessTool`).
3. Unified CTMC (`exp(Gt)` transition curves, stationary $\pi$, Gillespie jump trajectories, and `expected_absorption_times`) and bounded zero-allocation HMM (`baum_welch_with_workspace`, `viterbi`, and `posterior_probabilities`) regime-switching calibration (`MarkovRegimeTool`).
4. Plotting `expected_growth_rate` across stake fractions up to `F_DOMAIN_CAP = 0.99`—preserving visibility of the `2f*` ruin boundary even when net odds `b > 1`—with Quarter (`0.25 f*`), Half (`0.50 f*`), Full (`1.00 f*`), and `Double Kelly (2.00 f* Ruin)` markers plus `.json` sizing schedule exports in `BetSizeCalculatorTool`.

## Decision
1. **Zero-Allocation `pure_math` Stochastic Helpers**:
   - Extend `pure_math::statistics::copula` (`MAX_COPULA_DIMENSION = 8`) with `CorrelationMatrix::equicorrelated`, `GaussianCopula::multivariate_cdf`, and zero-allocation `GaussianCopula::sample_uniform_into`.
   - Add zero-allocation `EulerMaruyama::simulate_into<R: Rng>(&self, x0: f64, out: &mut [f64], rng: &mut R)` to `pure_math::statistics::ou_process`.
   - Add `MAX_BAUM_WELCH_ITERATIONS = 1_000` and zero-allocation `HiddenMarkovModel::baum_welch_with_workspace` / `HiddenMarkovModel::baum_welch` to `pure_math::statistics::markov::hmm`, verified under `VerifiedEngine::run_verified` in `crates/pure_math/tests/test_risk_zero_alloc.rs`.
2. **Interactive Risk Workbenches in `FinancialMathTab` (`math_explorer_gui`)**:
   - Register `CopulaRiskTool` (`"Copula Tail Risk"`), `OuProcessTool` (`"Ornstein-Uhlenbeck Calibration"`), and `MarkovRegimeTool` (`"Markov Regime Switching"`) inside `FinancialMathTab` (`ExportCapabilities::plot()`), alongside `StarterPresetMetadata` cards, `bounded_drag_value` parameter guards, `SemanticPalette` plot styling, UTF-8 BOM CSV import (`web_io::strip_utf8_bom`), and `MAX_DECODED_URL_STATE_BYTES` URL share warnings.
   - Upgrade `BetSizeCalculatorTool` with `F_DOMAIN_CAP = 0.99`, the `Double Kelly (2.00 f* Ruin)` point series, and `.json` sizing schedule copy/download exports routed through a shared `#[inline(never)]` `format_pretty_json` helper to minimize WASM code size.

## Consequences
- **Positive**: Users can price correlated `2..=8`-leg parlays, calibrate Ornstein-Uhlenbeck mean-reverting series from CSV files, fit CTMC/HMM regime models, and inspect fractional Kelly growth vs. `2f*` ruin boundaries interactively on native and WASM builds while preserving NASA Power of 10 Rule 3 zero-allocation invariants in `pure_math`.
- **Neutral**: Workspace GUI contract suites now track 89 registered `InteractiveTool` implementations (`68` plot dataset+figure + `7` raster + `14` non-figure; `86` non-shader + `3` shader) and 141 `StarterPresetMetadata` cards.
