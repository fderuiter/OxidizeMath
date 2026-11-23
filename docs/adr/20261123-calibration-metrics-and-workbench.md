# ADR 0047: Probabilistic Calibration Metrics, Post-Hoc Recalibrators, and Interactive Workbench

## Status

Accepted (2026-11-23)

## Context

Previously, workspace crates lacked shared proper scoring rules, reliability diagnostics, threshold-sweep ranking curves, and supervised post-hoc probability recalibrators:

- [`crates/domain_ai/src/ai/self_calibration/mod.rs`](../../crates/domain_ai/src/ai/self_calibration/mod.rs) implemented label-free LLM answer-clustering self-consistency (`self_calibration_paper.tex`) rather than supervised forecast calibration against ground-truth labels, and [`traceability.toml`](../../traceability.toml) mapped `domain_physics` to `self_calibration_paper.tex`.
- [`crates/domain_ai/src/ai/optimization.rs`](../../crates/domain_ai/src/ai/optimization.rs) (`cross_entropy_loss`) and [`crates/domain_ai/src/ai/deep_learning_theory/cycle.rs`](../../crates/domain_ai/src/ai/deep_learning_theory/cycle.rs) (`TrainingLoop::train_step`) clipped only the lower probability bound via `p.max(1e-15)` without symmetric `[epsilon, 1.0 - epsilon]` clamping, binary `1 - p` complement terms, or explicit `NaN` / dimension guards.
- [`crates/domain_climate/src/climate/loss.rs`](../../crates/domain_climate/src/climate/loss.rs) and [`crates/domain_applied/src/applied/clinical_trials/analysis.rs`](../../crates/domain_applied/src/applied/clinical_trials/analysis.rs) had no zero-allocation ensemble `CRPS` or prognostic calibration diagnostics, and [`math_explorer_gui/src/tabs/ai/mod.rs`](../../math_explorer_gui/src/tabs/ai/mod.rs) offered no interactive tool for inspecting reliability diagrams, Murphy decompositions, `ROC`/`PR` curves, or post-hoc recalibration.

## Decision

1. **Zero-Allocation Calibration, Scoring, Ranking, and Recalibration Suite in `pure_math`:**
   - Added [`crates/pure_math/src/pure_math/statistics/calibration/mod.rs`](../../crates/pure_math/src/pure_math/statistics/calibration/mod.rs) exposing `scoring` (`brier_score`, exact and binned `murphy_decomposition`, ULP-safe symmetric `binary_log_loss` and `multiclass_log_loss`, `statrs`-backed `gaussian_crps`, and shift-invariant zero-allocation `ensemble_crps_sorted`), `diagnostics` (`BinningStrategy::Uniform` / `Quantile`, `reliability_curve_uniform_into`, `expected_calibration_error`, `maximum_calibration_error`), `ranking` (`roc_curve`, `roc_auc`, `precision_recall_curve`, step and trapezoidal `pr_auc`), `recalibration` (damped 2x2 Newton-Raphson `PlattScaler` and globally convex inverse-temperature `TemperatureScaler`), and `isotonic` (zero-allocation `fit_pava_sorted_into`, $O(\log M)$ piecewise-linear `predict_pava_blocks`, and boundary-smoothed `IsotonicCalibrator`).
   - Added `OptimizationError::DimensionMismatch` and `OptimizationError::InvalidInput` in [`crates/pure_math/src/pure_math/analysis/optimization.rs`](../../crates/pure_math/src/pure_math/analysis/optimization.rs).

2. **Cross-Entropy Hardening, Cross-Domain Adapters, and Traceability Separation:**
   - Updated `cross_entropy_loss` in [`crates/domain_ai/src/ai/optimization.rs`](../../crates/domain_ai/src/ai/optimization.rs) and `TrainingLoop::train_step` in [`crates/domain_ai/src/ai/deep_learning_theory/cycle.rs`](../../crates/domain_ai/src/ai/deep_learning_theory/cycle.rs) to reject `NaN`/non-finite inputs and dimension mismatches while applying symmetric `[epsilon, 1.0 - epsilon]` clipping and binary complement loss terms.
   - Added supervised calibration adapters in [`crates/domain_ai/src/ai/calibration.rs`](../../crates/domain_ai/src/ai/calibration.rs), [`crates/domain_ai/src/ai/self_calibration/temperature.rs`](../../crates/domain_ai/src/ai/self_calibration/temperature.rs), [`crates/domain_applied/src/applied/clinical_trials/analysis.rs`](../../crates/domain_applied/src/applied/clinical_trials/analysis.rs) (`evaluate_prognostic_calibration`), and [`crates/domain_climate/src/climate/loss.rs`](../../crates/domain_climate/src/climate/loss.rs) (`ensemble_crps_loss`), while separating label-free self-consistency from supervised calibration in [`crates/domain_ai/README.md`](../../crates/domain_ai/README.md) and [`traceability.toml`](../../traceability.toml).

3. **Interactive `CalibrationWorkbenchTool` in `math_explorer_gui`:**
   - Added [`math_explorer_gui/src/tabs/ai/calibration_workbench/mod.rs`](../../math_explorer_gui/src/tabs/ai/calibration_workbench/mod.rs) and registered `CalibrationWorkbenchTool` in [`math_explorer_gui/src/tabs/ai/mod.rs`](../../math_explorer_gui/src/tabs/ai/mod.rs) with reliability diagrams, Murphy decomposition bars, `ROC` & `PR` threshold-sweep overlays, Gaussian/ensemble `CRPS` plots, live Platt/Temperature/PAVA recalibration controls, three `StarterPresetMetadata` presets, `#[serde(deny_unknown_fields)]` bounded URL state, and `ExportCapabilities::plot()`.

## Consequences

- Hot-path calibration kernels (`brier_score`, `binary_log_loss`, `multiclass_log_loss`, `gaussian_crps`, `ensemble_crps_sorted`, `reliability_curve_uniform_into`, `fit_pava_sorted_into`, `predict_pava_blocks`, `PlattScaler::fit`, `TemperatureScaler::fit_binary`) execute with zero heap allocations under `VerifiedAllocator`.
- Exact Murphy decomposition satisfies `brier_score = reliability - resolution + uncertainty` within `TOLERANCE_STANDARD`, and post-hoc Platt, temperature, and PAVA recalibration measurably reduce held-out `ECE` and `log_loss`.
- AI, clinical-trial, and climate crates consume a single verified statistical foundation without duplicating calibration math or introducing inter-domain dependency cycles.
