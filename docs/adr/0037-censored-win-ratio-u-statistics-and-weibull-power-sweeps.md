# ADR 0037: Right-Censored Win-Ratio U-Statistics, Bivariate Weibull Power Sweeps, and Interactive Stress Tester

## Status
Accepted (2026-11-21)

## Context

`crates/domain_applied::applied::win_ratio` previously offered `UnmatchedPairing` ($n_1 \times n_2$ pairwise comparisons) paired with `calculate_statistics(wins, losses)`, which estimated variance using an independent-binomial standard error $\sqrt{p(1-p)/(N_W + N_L)}$. While valid for $n$ independent matched pairs (`MatchedPairing`), applying an independent-binomial denominator to $n_1 \times n_2$ unmatched cross-arm pairs treats dependent comparisons sharing the same patients as independent observations. Under a $50$-per-arm null trial ($2,500$ pairs), this underestimated the log-win-ratio standard error by $\approx 5.8\times$ and inflated nominal $\alpha = 0.05$ Type I error to $\approx 75\%$.

Additionally:

1. `HigherIsBetter` and `LowerIsBetter` compared raw `f64` scalars without censoring flags, preventing Pocock/Finkelstein-Schoenfeld handling of administrative follow-up horizons and random loss to follow-up.
2. `BivariateWeibullModel` exposed analytical survival/PDF formulas without deterministic inverse-CDF sampling or Monte Carlo power sweeps.
3. `math_explorer_gui` exposed no composite win-ratio tool under `tabs/clinical_trials/`, left `SampleSizeCalculatorTool` without analytical power plots (`ExportCapabilities::none()`), and left `try_estimate_hazard_ratio` unwired in `SurvivalAnalysisTool`.

## Decision

1. **Two-Sample U-Statistic Variance & Right-Censored Comparators (`domain_applied`)**:
   - Keep `calculate_statistics` unchanged for matched-pair workflows and baseline tests, while adding `evaluate_unmatched_into`, `evaluate_unmatched`, and `compute_unmatched_u_statistics` in `win_ratio::u_statistic`.
   - Project the pairwise win/loss kernels onto per-subject treatment and control marginal win/loss counts (`UnmatchedScratch`) using the Pocock (2012) / Bebu & Lachin (2016) two-sample U-statistic variance estimator for $\ln(\widehat{WR})$, alongside Dong et al. Win Odds and Buyse Net Treatment Benefit.
   - Introduce `CensoredOutcome`, `RightCensoredComparator`, and `TierCount` in `win_ratio::censoring` so observed events prior to common follow-up score wins/losses while indeterminate censored pairs return `ComparisonResult::Tie` and fall through to lower priority tiers.

2. **Deterministic Zero-Allocation Bivariate Weibull Power Sweeps (`domain_applied`)**:
   - Implement exact two-step inverse-CDF sampling (`try_sample_event_times`, `try_sample_censored_subject`) for $S(t, x) = \exp(-(\lambda_1 t + \lambda_2 x)^\alpha)$ driven strictly by `oxidize_core::rng::OxidizeRng` with explicit `u64` seeds and bounded Newton-Raphson inversion of the concave conditional hazard increment.
   - Preallocate cohort and U-statistic buffers in `PowerSweepWorkspace` so `simulate_weibull_trial_into` and `run_power_sweep_into` execute with zero dynamic heap allocations under `VerifiedAllocator`, enforcing interactive caps (`MAX_ARM_SIZE = 500`, `MAX_SWEEP_REPLICATES = 500`, `MAX_SWEEP_STEPS = 16`).

3. **Interactive GUI Stress Tester, Cohort Import, and Takeaway Exports (`math_explorer_gui`)**:
   - Register `WinRatioStressTesterTool` (`"Composite Win Ratio Stress Tester"`) with `egui_plot` Power-by-$N$ curves, hierarchical tier waterfalls, local `.csv`/`.json` cohort import via `web_io::spawn_open_file_dialog`, structured `.csv`/`.json` protocol exports via `web_io::spawn_save_as_file_dialog`, `ExportCapabilities::plot()`, compact URL state omitting uploaded patient rows, and a `StarterPresetMetadata` card.
   - Upgrade `SampleSizeCalculatorTool` to `ExportCapabilities::plot()` with analytical power curves (`calculate_power_means`, `calculate_power_proportions`) and wire two-arm Kaplan-Meier curves + `try_estimate_hazard_ratio` into `SurvivalAnalysisTool`.

## Consequences

- **Positive:** Empirical Type I error at $\alpha = 0.05$ across $2,000$ null $50$-per-arm trials is calibrated within $[0.035, 0.065]$; default interactive sweeps complete in tens of milliseconds identically across native desktop and `wasm32-unknown-unknown` with zero heap allocations inside hot loops; URLs remain compact within `URL_JSON_BOUNDS` ($4,096$ bytes).
- **Trade-offs:** Asymptotic log-linear U-statistic intervals require at least $2$ subjects per arm and fall back to net-benefit Z-tests when one arm records zero wins or zero losses. Adaptive group-sequential stopping boundaries and covariate-adjusted Cox models remain out of scope.
