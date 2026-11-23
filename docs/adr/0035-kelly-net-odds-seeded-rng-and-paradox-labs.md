# ADR 0035: Kelly Net-Odds Conversion, Growth-Rate Domain Guard, and Deterministic Seeded Reruns

## Status
Accepted (extends [ADR 0031](0031-calibrated-climate-series-and-conv-rng-seeding.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
Three mathematical and deterministic-simulation defects affected interactive stochastic and paradox laboratories across `pure_math`, `oxidize_core`, and `math_explorer_gui`:
1. **Net vs. Decimal Odds Mismatch in Kelly Criterion Callers**: [`Odds::new`](../../crates/pure_math/src/pure_math/statistics/kelly/core.rs) validates decimal odds strictly greater than `1.0`, whereas Kelly wagering sliders and literature frequently parameterize payoffs by net profit multiplier $b > 0$ (`decimal = 1 + b`). Passing net odds $b \le 1.0$ (such as even-money $b = 1.0$) directly to `Odds::new` failed validation, while paying wins with `Odds::value()` instead of `Odds::net_profit_multiplier()` overpaid every winning wager by `1x` stake.
2. **Premature Domain Cutoff in `expected_growth_rate`**: [`expected_growth_rate`](../../crates/pure_math/src/pure_math/statistics/kelly/criterion.rs) guarded $g(f) = p \ln(1 + b f) + q \ln(1 - f)$ with `f >= 1.0 / b` instead of checking `f >= 1.0 || 1.0 + b * f <= 0.0`, erroneously returning `f64::NEG_INFINITY` for valid fractions $f \in [1/b, 1)$ whenever $b > 1$.
3. **Constant-Seed Replay on Interactive Reruns**: [`OxidizeRng::default`](../../crates/oxidize_core/src/rng.rs) always initialized `ChaCha8Rng` with `0x0001_D1CE_CAFE_BABE`, causing interactive `"Rerun Simulation"` and `"Allocate Subjects"` actions to replay identical sample paths on every click.

## Decision
1. **Explicit `Odds::from_net_odds` Constructor**:
   - Add `Odds::from_net_odds(b: f64) -> Result<Odds, KellyError>` in [`crates/pure_math/src/pure_math/statistics/kelly/core.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/core.rs), validating `b.is_finite() && b > 0.0` and delegating to `Odds::new(1.0 + b)`.
   - Update [`BankrollGrowthTool`](../../math_explorer_gui/src/tabs/financial_math/bankroll_growth.rs) to construct `Odds::from_net_odds(self.odds)` and settle winning bets with `bet_amount * o.net_profit_multiplier()`.
2. **Exact Logarithmic Domain Guard in `expected_growth_rate`**:
   - Update [`expected_growth_rate`](../../crates/pure_math/src/pure_math/statistics/kelly/criterion.rs) to return `f64::NEG_INFINITY` only when `f >= 1.0 || f < 0.0 || 1.0 + b * f <= 0.0`, enabling continuous volatility-drag evaluation across $f \in [0, 0.95]$ in [`bankroll_ensemble.rs`](../../math_explorer_gui/src/tabs/financial_math/bankroll_ensemble.rs).
3. **Deterministic `BASE_SEED` and `OxidizeRng::from_seed` Streams**:
   - Export `BASE_SEED` and `OxidizeRng::from_seed(seed: u64)` in [`crates/oxidize_core/src/rng.rs`](../../crates/oxidize_core/src/rng.rs) so interactive tools advance a deterministic `seed_counter: u64` (`OxidizeRng::from_seed(BASE_SEED ^ seed_counter)`) without invoking prohibited OS entropy sources.
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) to record the expanded `Odds` and `OxidizeRng` public signatures.

## Consequences
- **Positive**: Net-odds inputs ($b \in [0.1, 10.0]$), winning bankroll settlements, and logarithmic growth rates $g(f)$ match Kelly (1956) analytically across sub-Kelly, full-Kelly, and overbetting ($f > 2f^*$) regimes.
- **Positive**: Interactive reruns in financial and clinical-trial tools generate distinct, cross-platform reproducible `ChaCha8Rng` realizations while continuing to pass `check-entropy` and `VerifiedAllocator` zero-allocation invariants.
