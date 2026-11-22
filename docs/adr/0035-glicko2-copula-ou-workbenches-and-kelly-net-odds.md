# ADR 0035: Glicko-2 Tournament, Correlated Risk & OU Workbenches, and Kelly Net-Odds Semantics

## Status
Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/pure_math/README.md`](../../crates/pure_math/README.md) positions the statistical modeling modules in `pure_math`—specifically Glicko-2 competitive ratings, Gaussian copulas, Ornstein-Uhlenbeck mean-reverting stochastic processes, and Kelly criterion capital allocation—as reference engines for interactive `math_explorer_gui` visualizers. Three gaps prevented end-to-end competitive and quantitative risk workflows in the GUI:

1. **Unexposed Competitive Rating, Copula, and OU Engines**: [`crates/pure_math/src/pure_math/statistics/glicko2/rating.rs`](../../crates/pure_math/src/pure_math/statistics/glicko2/rating.rs), [`crates/pure_math/src/pure_math/statistics/copula/gaussian.rs`](../../crates/pure_math/src/pure_math/statistics/copula/gaussian.rs), and [`crates/pure_math/src/pure_math/statistics/ou_process/solver.rs`](../../crates/pure_math/src/pure_math/statistics/ou_process/solver.rs) implemented verified solvers (`update_rating`, `increase_rd_for_inactivity`, `expected_outcome`, `GaussianCopula::bivariate`, `sgp_joint_probability`, and `EulerMaruyama`) without interactive workbenches in [`math_explorer_gui/src/tabs/game_theory/mod.rs`](../../math_explorer_gui/src/tabs/game_theory/mod.rs) or [`math_explorer_gui/src/tabs/financial_math/mod.rs`](../../math_explorer_gui/src/tabs/financial_math/mod.rs).
2. **Net-Odds vs. Decimal-Odds Mismatch in `BankrollGrowthTool`**: [`math_explorer_gui/src/tabs/financial_math/bankroll_growth.rs`](../../math_explorer_gui/src/tabs/financial_math/bankroll_growth.rs) labeled its slider `Net Odds (b)` (`b = 1.0` for even money), yet validated `self.odds` via decimal `Odds::new` (`> 1.0`, rejecting `b = 1.0`) and credited wins with `bet_amount * o.value()` instead of `bet_amount * o.net_profit_multiplier()`, doubling profit at `2.0`.
3. **Missing Growth Curve Visualization and Exports in `BetSizeCalculatorTool`**: [`math_explorer_gui/src/tabs/financial_math/bet_size.rs`](../../math_explorer_gui/src/tabs/financial_math/bet_size.rs) returned `ExportCapabilities::none()` and omitted `kelly::expected_growth_rate` curves and copyable/downloadable sizing tables.

## Decision
1. **Add `Odds::from_net_odds` in `pure_math` and Fix `BankrollGrowthTool` Compounding**:
   - Introduce `#[verified_engine::verified] pub fn from_net_odds(net_odds: f64) -> Result<Self, KellyError>` in [`crates/pure_math/src/pure_math/statistics/kelly/core.rs`](../../crates/pure_math/src/pure_math/statistics/kelly/core.rs), validating finite `net_odds > 0.0` and converting to decimal odds `1.0 + net_odds`.
   - Update [`math_explorer_gui/src/tabs/financial_math/bankroll_growth.rs`](../../math_explorer_gui/src/tabs/financial_math/bankroll_growth.rs) (`Default`, `recalculate`, `try_load_state`, `TheoryDescribable`, and `STARTER_PRESET_JSON`) to validate `Odds::from_net_odds` (default `1.0` even money) and credit winning wagers with `bet_amount * o.net_profit_multiplier()`.
2. **Enable Cached Growth Curves and Formula-Safe Table Exports in `BetSizeCalculatorTool`**:
   - Upgrade `BetSizeCalculatorTool` to `ExportCapabilities::plot()`, caching `expected_growth_rate` `PlotPoint` curves and Quarter (`0.25 f*`), Half (`0.50 f*`), and Full (`f*`) Kelly markers with borrowed slice rendering (`PlotPoints::Borrowed`).
   - Add formula-safe CSV (`egui_plot::export::escape_csv_field`) and Markdown Kelly sizing table copy and save-as actions via [`math_explorer_gui/src/web_io.rs`](../../math_explorer_gui/src/web_io.rs).
3. **Add `Glicko2TournamentTool` (`GameTheoryTab`) and `CorrelatedRiskScenarioTool` (`FinancialMathTab`)**:
   - Wire `Glicko2TournamentTool` into `GameTheoryTab` with frozen pre-period roster snapshots so match order never skews simultaneous `update_rating` calls, automatic `increase_rd_for_inactivity` updates for zero-match players, cached `r +/- 2 * RD` 95% skill-interval error bars, pairwise `expected_outcome` win-probability matrices, bounded client-side CSV match imports (`<= 64` players, `<= 1,024` matches, `<= 64 KiB`), and CSV/Markdown exports.
   - Wire `CorrelatedRiskScenarioTool` into `FinancialMathTab` plotting bivariate `GaussianCopula::bivariate` and `sgp_joint_probability` curves across `rho` in `[-0.99, 0.99]` against naive independent `p_a * p_b`, alongside deterministic `EulerMaruyama` Monte Carlo Ornstein-Uhlenbeck paths seeded via `oxidize_core::rng::OxidizeRng::new` (`<= 200` paths, `<= 2,000` steps).

```text
Updated GUI inventory totals:
- InteractiveTools: 57 (35 plot + 7 raster + 15 non-export)
- StarterPresetMetadata cards: 61
```

## Consequences
- **Positive**: Users can analyze simultaneous Glicko-2 tournaments, bivariate tail dependence, mean-reverting OU trajectories, and accurate net-odds Kelly compounding directly in the browser with full PNG, SVG, CSV, JSON, Markdown, and `#tab=...` URL share support.
- **Negative / Trade-offs**: Adding `Odds::from_net_odds` extends the public `pure_math` and `math_explorer` API surface, requiring `api_baseline.json` regeneration. Splitting tool logic, UI, and export formatters into focused submodules preserves the 500-line file limit, 60-line function limit, zero-allocation steady-state frames, and `< 3 MB` WASM bundle budget.
