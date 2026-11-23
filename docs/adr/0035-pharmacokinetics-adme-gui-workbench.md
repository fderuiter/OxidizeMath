# ADR 0035: Pharmacokinetics (ADME) Regimen Workbench in `ClinicalTrialsTab`

## Status
Accepted (extends [ADR 0016](0016-standardize-crate-documentation-suite.md), [ADR 0033](0033-wasm32-browser-tests-and-playwright-ci-smoke-gate.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/domain_applied/src/applied/pharmacokinetics/`](../../crates/domain_applied/src/applied/pharmacokinetics/mod.rs) implements verified one-compartment extravascular ADME models (`PKParameters`, `BatemanModel`, `half_life`, `t_max`, `solve_ka`, `SuperpositionModel`, `TwoPulseModel`, and `EnantiomerModel`), re-exported through `math_explorer::applied::pharmacokinetics`. However, `ClinicalTrialsTab` in [`math_explorer_gui`](../../math_explorer_gui/src/tabs/clinical_trials/mod.rs) exposed only `SampleSizeTool` and `SurvivalAnalysisTool`, leaving clinical pharmacometricians without an interactive multi-dose ADME regimen planner, therapeutic window evaluator, or facade-canonical documentation example in [`crates/domain_applied/README.md`](../../crates/domain_applied/README.md).

Two architectural constraints govern integrating `pharmacokinetics` into `math_explorer_gui`:
1. **Zero-Allocation Sampling Under `VerifiedAllocator`**: `EnantiomerModel::concentration_{ir,xr}_multiple_doses` clones `dose_times` into a temporary `Vec<f64>` per call. Invoking those convenience helpers inside a time-sampling loop would allocate on the heap per sample point and trip `VerifiedAllocator`.
2. **Convexity-Safe `solve_ka` Root Calibration**: Newton-Raphson calibration via `solve_ka` requires a strictly positive initial guess that converges reliably across physiological `T_max` and `k_e` combinations while surfacing `PharmacokineticsError` diagnostics cleanly without panicking when an infeasible target is requested.

## Decision
1. **Register `PharmacokineticsTool` and `pharmacokinetics_adme_starter` in `ClinicalTrialsTab`**:
   - Add [`math_explorer_gui/src/tabs/clinical_trials/pharmacokinetics/`](../../math_explorer_gui/src/tabs/clinical_trials/pharmacokinetics/mod.rs) under `@explorer_feature = "applied"`, registering `ToolMetadata` (`id: "PharmacokineticsTool"`, `display_name: "Pharmacokinetics (ADME)"`, `domain: "clinical_trials"`) and `StarterPresetMetadata` (`id: "pharmacokinetics_adme_starter"`).
   - Support all four compositional formulations—single-entity immediate release (`BatemanModel`), single-entity biphasic extended release (`TwoPulseModel<BatemanModel>`), chiral immediate release (`EnantiomerModel`), and chiral biphasic extended release (`TwoPulseModel<EnantiomerModel>`)—over multi-dose schedules (`1..=64` slots) with optional initial loading doses and toggleable missed-dose slots.
2. **Separate Model Compilation from Zero-Allocation Curve & Metric Evaluation**:
   - Construct `SuperpositionModel` instances and reserve `PlotPoint` buffers (`<= 2,000` points) once per parameter update in `compile_models()`.
   - Evaluate total and per-isomer (`d`/`l`) plasma concentration curves alongside all eight PK metrics (`C_max`, `C_min`, `T_1/2`, `T_max`, `CL`, trapezoidal `AUC`, steady-state `R_ac`, and `%` time inside `[MEC, MTC]`) in `evaluate_sampled_regimen()` with zero dynamic heap allocations, verified under `VerifiedEngine::run_verified`.
   - Use convexity of $T_{\max}(k_a, k_e)$ to select a bracket-safe lower initial guess $0 < k_{a,0} \le k_a^*$ before invoking `solve_ka`, rendering any `PharmacokineticsError` via `SemanticPalette::error`.
3. **Provide Multi-Format Exports, Bounded Persistence, and Facade Documentation**:
   - Enable `ExportCapabilities::plot()`, `TraceSnapshotManager` overlays, a base-10 logarithmic Y-axis projection toggle, and clipboard/file export of Markdown (`.md`) and CSV (`.csv`) dosing summary cards via `web_io::spawn_save_as_file_dialog`.
   - Enforce `ToolStateBounds::DEFAULT` and `#[serde(deny_unknown_fields)]` across `.mws` workspaces, recovery snapshots, and `#tab=clinical_trials&tool=PharmacokineticsTool&state=...` share links, and document multi-dose ADME usage in [`crates/domain_applied/README.md`](../../crates/domain_applied/README.md) using `math_explorer::applied::pharmacokinetics`.

## Consequences
- **Positive**: Users can interactively plan immediate-release, extended-release, and chiral dosing regimens, inspect `[MEC, MTC]` therapeutic window coverage on linear or `log10` axes, overlay pinned trace snapshots, and export figures, curves, and `.md`/`.csv` dosing cards without mutating any `crates/` public API signatures.
- **Positive**: Pre-compiling `SuperpositionModel` instances outside `evaluate_sampled_regimen()` preserves strict zero-allocation guarantees under `VerifiedAllocator` while maintaining exact analytical parity with `domain_applied::applied::pharmacokinetics`.
- **Neutral**: Registered interactive tool and starter preset contract counts in `math_explorer_gui` increase from `55 -> 56` tools (`32 -> 33` plot-export tools) and `59 -> 60` starter presets.
