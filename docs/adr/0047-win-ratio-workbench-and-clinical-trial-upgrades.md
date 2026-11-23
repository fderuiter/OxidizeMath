# ADR 0047: Composite Win Ratio Workbench and Clinical Trial Biostatistics Upgrades

## Status
Accepted (extends [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), [ADR 0028](0028-finite-survival-time-and-total-order-kaplan-meier.md), and [ADR 0035](0035-pharmacokinetics-adme-gui-workbench.md))

## Context
[`crates/domain_applied/src/applied/win_ratio/`](../../crates/domain_applied/src/applied/win_ratio/mod.rs) implements a comprehensive prioritized composite endpoint suite—`WinRatioAnalysis`, `HigherIsBetter`, `LowerIsBetter`, `ThresholdComparator`, `UnmatchedPairing`, `MatchedPairing`, Pocock two-tier `WinLossCounts` (`calculate_sample_win_ratio`, `calculate_win_proportion`, `calculate_confidence_interval`, `calculate_significance_test_statistic`), `calculate_statistics` (`WinRatioStats`), `BivariateWeibullModel` + Clenshaw-Curtis `ProbabilityWinRatioContext`, and anthropometric `calculate_bmi`—yet had zero interactive GUI exposure in [`math_explorer_gui/src/tabs/clinical_trials/`](../../math_explorer_gui/src/tabs/clinical_trials/mod.rs).

At the same time, existing biostatistical tools in `ClinicalTrialsTab` exhibited four workflow gaps:
1. [`SurvivalAnalysisTool`](../../math_explorer_gui/src/tabs/clinical_trials/survival/mod.rs) accepted only single-arm textarea strings, leaving `try_estimate_hazard_ratio`, `TimePoint` risk-set counts (`n_at_risk`, `n_events`, `n_censored`), local CSV/TSV cohort import, and `.csv`/`.md`/`.svg` exports unexposed.
2. [`SampleSizeCalculatorTool`](../../math_explorer_gui/src/tabs/clinical_trials/sample_size/mod.rs) returned `ExportCapabilities::none()` and rendered only a scalar count without a statistical power curve ($1 - \beta$ vs $N$) or protocol sizing tables.
3. [`RandomizationTool`](../../math_explorer_gui/src/tabs/clinical_trials/randomization.rs) and [`SimpsonParadoxLab`](../../math_explorer_gui/src/tabs/clinical_trials/simpson_paradox.rs) discarded per-patient `assigned` IDs from `StratifiedRandomizer::assign_stratified`, disabled `ExportCapabilities::plot()` in `Simple` and `Block` modes, and lacked `.csv`/`.json` allocation schedule downloads.
4. Clinical biostatisticians require 100% client-side, HIPAA-safe in-memory CSV/TSV file ingestion and report generation with bounded row counts (`MAX_ITERATIONS = 5,000`) and zero heap allocations in numeric hot paths under `VerifiedAllocator`.

## Decision
1. **Add `WinRatioTool` and Shared `ClinicalIoState` in `ClinicalTrialsTab`**:
   - Add [`math_explorer_gui/src/tabs/clinical_trials/win_ratio/`](../../math_explorer_gui/src/tabs/clinical_trials/win_ratio/mod.rs) (`ToolMetadata` `id: "WinRatioTool"`, `display_name: "Composite Win Ratio"`) and two `StarterPresetMetadata` cards (`win_ratio_composite_starter`, `win_ratio_weibull_prc_starter`), backed by [`clinical_io.rs`](../../math_explorer_gui/src/tabs/clinical_trials/clinical_io.rs) wrapping `web_io::spawn_open_file_dialog` and `web_io::spawn_save_as_file_dialog`.
   - Separate workspace buffer preparation (`prepare_workspace()`) from zero-allocation numerical evaluation (`evaluate_hot_path()`), running `WinRatioAnalysis`, per-tier cumulative attribution, Pocock `WinLossCounts`, secondary continuous `TTestIndependent` (`GroupData`), `calculate_bmi`, and Clenshaw-Curtis `ProbabilityWinRatioContext` (`calculate_win_probability`, `calculate_loss_probability`, `calculate_probability_win_ratio`) over `BivariateWeibullModel` horizons $c \in [c_{\min}, c_{\max}]$ ($c_{\min} > 0$ to avoid the $t^{\alpha - 1}$ singularity when $\alpha < 1$).
2. **Upgrade `SurvivalAnalysisTool`, `SampleSizeCalculatorTool`, and `RandomizationTool`**:
   - Upgrade [`survival/`](../../math_explorer_gui/src/tabs/clinical_trials/survival/mod.rs) to parse single-arm and multi-arm CSV/TSV cohorts up to `MAX_ITERATIONS`, preserve legacy `SurvivalState { input_text }` compatibility, compute two-arm `try_estimate_hazard_ratio`, tabulate `TimePoint` risk sets, and export `.csv`, `.md`, and `.svg` artifacts.
   - Upgrade [`sample_size/`](../../math_explorer_gui/src/tabs/clinical_trials/sample_size/mod.rs) to sweep $1 - \beta \in [0.50, 0.99]$, advertise `ExportCapabilities::plot()`, and export `.csv` and `.md` protocol sizing schedules without lint suppressions.
   - Add [`randomization_schedule.rs`](../../math_explorer_gui/src/tabs/clinical_trials/randomization_schedule.rs) (`SubjectAllocationRecord`) so `RandomizationTool` and `SimpsonParadoxLab` retain per-patient IDs (`SUBJ-0001..`, `M-i`, `S-i`), plot cumulative arm balance (`ExportCapabilities::plot()` across `Simple`, `Block`, and `Stratified`), and export `.csv` and `.json` schedules.

## Consequences
- **Positive**: Every clinical trial tool in `ClinicalTrialsTab` now advertises `ExportCapabilities::plot()`, passes offscreen PNG/SVG modal audits, executes 100% locally in memory on desktop and `wasm32-unknown-unknown`, and exposes end-to-end report/table downloads.
- **Positive**: `WinRatioTool::evaluate_hot_path` executes with zero heap allocations under `VerifiedEngine::run_verified`, and `MatchedPairing` enforces equal arm lengths prior to evaluation so domain panics are impossible from user input.
- **Neutral**: Global GUI registry contracts advance from `75 -> 76` `InteractiveTool` entries (`52 -> 55` plot-export tools, `16 -> 14` non-figure tools) and `108 -> 110` `StarterPresetMetadata` cards.
