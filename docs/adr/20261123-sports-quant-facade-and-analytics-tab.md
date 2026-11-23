# ADR 20261123: Sports Quant Facade, Traceability Manuscript, and Sports Analytics Tab

## Status

Accepted

## Context

OxidizeMath implemented five quantitative sports and wagering sub-modules inside Layer 3 (`crates/pure_math/src/pure_math/statistics`: `kelly`, `copula`, `glicko2`, `zip_regression`, and `ou_process`), yet three architectural gaps limited usability and theoretical traceability:

1. **Fragmented probability wrappers**: `copula::Probability`, `kelly::BankrollFraction`, and `zip_regression::ZeroInflation` each wrapped `[0, 1]` scalars without `From<UnitInterval>` / `From<T> for UnitInterval` bridges, forcing callers to unwrap raw `f64` values and re-validate invariants when piping model probabilities into Kelly sizing.
2. **Missing companion pricing kernels and facade**: Standard sportsbook vigorish removal (Multiplicative, Additive, Power, and Shin insider equilibrium), Elo margin-of-victory ratings, and bivariate match score matrices (Independent Poisson, Dixon-Coles, Karlis-Ntzoufras) were absent, and sports modules were buried four namespaces deep rather than exposed at a cohesive `math_explorer::sports_quant` surface.
3. **Misattributed citations and split GUI tools**: Wagering and rating modules carried placeholder `[cite:clinical_trials_statistics]` and `[cite:modular_polynomials_review]` annotations rather than a dedicated sports-quant manuscript, and five core pricing/dependence/prop/in-play modules lacked dedicated interactive workbench tools under `Applied Math`.

At the same time, OxidizeMath's `xtask test-features` matrix enforces $2^N$ combinatorial builds across the exact seven Cargo features (`ai`, `applied`, `biology`, `climate`, `epidemiology`, `physics`, `pure_math`), making a new Layer 4 crate or eighth Cargo feature undesirable.

## Decision

1. **Preserve the 7-feature architecture**: Implement companion zero-allocation kernels (`market_pricing`, `elo`, `match_models`, and `kelly::ruin`) inside Layer 3 `pure_math::statistics` and expose a curated Layer 5 `#[cfg(feature = "pure_math")] pub mod sports_quant` facade plus `prelude` re-exports in `math_explorer`.
2. **Infallible `UnitInterval` probability bridges**: Implement zero-cost bidirectional `From<UnitInterval>` conversions on `Probability`, `BankrollFraction`, and `ZeroInflation`, alongside `#[verified]` `UnitInterval`-returning accessors (`Odds::implied_probability_unit`, `ZipDistribution::{pmf_unit, over_probability_unit}`, `OuAnalyzer::comeback_probability_unit`, and `head_to_head_win_probability_unit`).
3. **Dedicated manuscript and traceability repair**: Author `papers/sports_quantitative_modeling.tex` with canonical BibTeX entries (`kelly1956new`, `thorp1969optimal`, `shin1993measuring`, `dixon1997modelling`, `karlis2003analysis`, `lambert1992zero`, `glickman1999parameter`, `elo1978rating`), map all wagering/rating modules in `traceability.toml`, and wire `math_explorer_gui/src/framework/theory_links.rs` so the Theory Context Portal links directly to `sports_quantitative_modeling.pdf` and DOI references.
4. **`SportsAnalyticsTab` under `Applied Math`**: Add `math_explorer_gui/src/tabs/sports_analytics/` (`// @explorer_feature = "pure_math"`, `academic_domain() -> "Applied Math"`, `url_slug() -> "sports_analytics"`) hosting seven `ExportCapabilities::plot()` tools (`MarketDeviggingTool`, `MatchScoreHeatmapTool`, `SgpCopulaTool`, `ZipCountPropsTool`, `LiveMomentumTool`, `SportsRatingsTool`, `ClvRuinTool`) with `SemanticPalette` styling, `URL_JSON_BOUNDS` deep links, and `StarterPresetMetadata` cards, while upgrading `BetSizeCalculatorTool` with Kelly log-growth plot and dataset exports.

## Consequences

- Callers compose devigging, match score grids, ZIP props, Gaussian copula SGPs, OU comeback probabilities, and Elo/Glicko-2 ratings directly into Kelly sizing without `f64` unwrapping.
- Combinatorial feature build time remains unchanged at $2^7 = 128$ configurations, and compressed WASM bundles remain well under the 3 MiB (`3,145,728` byte) gate.
- Bidirectional theory-to-code parity increases to 30 compiled LaTeX manuscripts with zero orphaned papers and verified BibTeX DOI links across all sports and financial wagering tools.
