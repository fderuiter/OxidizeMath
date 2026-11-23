# ADR 0044: Zero-Allocation Brent Root-Finder, Market Pricing Devigging, and Execution Calculators

## Status
Accepted (extends [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), [ADR 0035](0035-kelly-net-odds-seeded-rng-and-paradox-labs.md), and [ADR 0042](0042-hierarchical-umbrella-errors-and-diagnostic-bus-helpers.md))

## Context
`pure_math::statistics::kelly` previously assumed callers already knew true win probabilities `UnitInterval`, while `pure_math::analysis::roots` (`Bisection` and `NewtonRaphson`) formatted `String` error messages inside `AnalysisError::InvalidParameters(String)`—triggering heap allocations under `VerifiedEngine::run_verified`—and lacked a superlinear derivative-free bracketed solver (`Brent`). As a result, users could not remove bookmaker overround ("vig") under favorite-longshot bias (`Power` and `Shin`), detect negative probabilities when `Additive` vig removal breaches the probability simplex on longshots, or size cross-book arbitrage, subset Dutching, spread/total middles, and live/futures hedges in `FinancialMathTab`.

## Decision
1. **Zero-Allocation `AnalysisError` and `Brent` Solver (`pure_math::analysis::roots`)**:
   - Replace `AnalysisError::InvalidParameters(String)` and runtime `format!` calls with `Copy` scalar variants (`ConvergenceError(f64)`, `RootNotBracketed { min, f_min, max, f_max }`, `DerivativeTooSmall { x, derivative }`, and `InvalidBracket { min, max }`).
   - Implement `Brent` (`RootFinder`) combining inverse quadratic interpolation, secant steps, and guaranteed bisection fallback within zero-allocation `#[verified_engine::verified]` bounds.
2. **Slice-Based `pure_math::statistics::market_pricing` and `MarketPricingError`**:
   - Add zero-allocation `&[Odds]` routines bounded by `MAX_MARKET_OUTCOMES = 64` for `booksum`, `overround`, `turnover_margin`, `Multiplicative`, `Additive` (emitting `SimplexViolationDiagnostic` alongside raw and simplex-clamped probabilities when longshots fall `<= 0.0`), `Power` (dynamic analytical upper bracket solved via `Brent`), and `Shin` (closed-form 2-way `z` and rationalized `N`-way `Brent` solver on `[0.0, 1.0]`).
   - Pair devigged fair probabilities with `kelly_fraction` (`evaluate_outcome_ev`, `compare_outcome_ev_all_methods`) and add `evaluate_arbitrage_into`, `dutch_outcomes_into`, `dutch_subset_into`, `evaluate_middle`, and `evaluate_hedge`.
   - Expose `Copy` `MarketPricingError` through `PureMathError` and `math_explorer::error::Error`.
3. **Interactive `FinancialMathTab` Calculators (`math_explorer_gui`)**:
   - Register `MarketDeviggingTool` (`"Fair Odds & +EV Devigger"`) and `MarketExecutionTool` (`"Arbitrage, Dutching & Hedge"`) under `financial_math` with Starter Gallery presets, `#[serde(deny_unknown_fields)]` URL state round-trips, `SemanticPalette` tokens, inline Additive simplex-violation banners, and `egui_plot` series.

## Consequences
- **Positive**: `Bisection`, `NewtonRaphson`, `Brent`, and every `market_pricing` routine execute with zero heap allocations inside `VerifiedEngine::run_verified` while matching analytical devigging, `+EV`, arbitrage, Dutching, middle, and hedge benchmarks within `1e-9`.
- **Neutral**: Workspace contract suites now track 28 crate error enums (50 leaf errors), 72 registered `InteractiveTool` implementations (49 plot-exportable tools), and 103 `StarterPresetMetadata` cards.
