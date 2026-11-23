# ADR 0043: Zero-Allocation Variable-Dose Pharmacokinetic Regimen Superposition and Interactive Planner

## Status
Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0023](0023-separate-metadata-ids-from-display-labels.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/domain_applied/src/applied/pharmacokinetics/mod.rs`](../../crates/domain_applied/src/applied/pharmacokinetics/mod.rs) previously implemented one-compartment Bateman kinetics (`BatemanModel`), chiral mixtures (`EnantiomerModel`), biphasic extended-release pulses (`TwoPulseModel`), and uniform-dose superposition (`SuperpositionModel`), but had four gaps preventing end-to-end multi-dose and adherence analysis:
1. **Uniform-Dose Superposition Limitation**: `SuperpositionModel` evaluated a single fixed `base_model` across `dose_times`, preventing variable per-event schedules such as `2D` loading doses, missed doses followed by fractional `1.5D` catch-up doses, or mixed immediate-release (IR) and extended-release (XR) boosters.
2. **Missing Exposure and Therapeutic-Window Analytics**: Callers had no domain primitives to compute peak (`C_max`), terminal-interval trough (`C_min`), trapezoidal exposure (`AUC_total`, `AUC_0_tau`), theoretical and empirical accumulation ratios (`R_ac`), Time-in-Therapeutic-Range (`%` and hours), or supratherapeutic toxic time (`%` and hours) over preallocated slices inside locked `VerifiedAllocator` regions.
3. **Absent Interactive GUI Surface**: No `InteractiveTool` exposed `domain_applied::applied::pharmacokinetics` inside [`math_explorer_gui/src/tabs/medical/mod.rs`](../../math_explorer_gui/src/tabs/medical/mod.rs), leaving absorption-rate calibration (`solve_ka`), therapeutic window visualization, URL state sharing, and CSV/JSON takeaway exports inaccessible to interactive users.
4. **Traceability Misalignment**: [`traceability.toml`](../../traceability.toml) mapped `pharmacokinetics` to `quantum_mechanics.tex` rather than `clinical_trials_statistics.tex`.

## Decision
1. **Stack-Bounded Variable-Dose Superposition (`DoseEvent` & `RegimenSuperposition`)**:
   - Add [`crates/domain_applied/src/applied/pharmacokinetics/regimen.rs`](../../crates/domain_applied/src/applied/pharmacokinetics/regimen.rs) defining `MAX_REGIMEN_EVENTS = 64`, `ReleaseFormulation` (`ImmediateRelease` and `ExtendedRelease { lag_time_h, f1, f2 }`), `DoseEvent { time_h, dose_mg, formulation }`, `RegimenModel`, and `RegimenSuperposition<M>` alongside `SuperpositionModel`.
   - Pre-scale per-event `Copy` models (`BatemanModel`, `EnantiomerModel`, and `TwoPulseModel`) via `PKParameters::with_dose` into fixed-capacity stack arrays (`[M; MAX_REGIMEN_EVENTS]`) during `RegimenSuperposition::new` so `concentration`, `sample_into`, and `sample_isomers_into` execute with **zero heap allocations** inside `VerifiedEngine::run_verified`.
   - Expose `calibrate_ka_from_half_life` in [`crates/domain_applied/src/applied/pharmacokinetics/bateman.rs`](../../crates/domain_applied/src/applied/pharmacokinetics/bateman.rs) to select a globally convergent initial guess for `solve_ka` from target $T_{\max}$ and $t_{1/2}$.
2. **Zero-Allocation Exposure and Window Analytics**:
   - Add [`crates/domain_applied/src/applied/pharmacokinetics/analytics.rs`](../../crates/domain_applied/src/applied/pharmacokinetics/analytics.rs) implementing `TherapeuticWindow`, `ExposureSummary`, `peak_concentration`, `trough_concentration`, `trapezoidal_auc`, `auc_interval`, `accumulation_ratio`, and `summarize_exposure` directly over caller slices with zero heap allocations.
3. **`MedicalTab` Planner, `URL_JSON_BOUNDS` Share Links, and Takeaway Artifacts**:
   - Register `PharmacokineticRegimenTool` (`display_name: "Pharmacokinetic Regimen"`) and three `StarterPresetMetadata` cards under [`math_explorer_gui/src/tabs/medical/pharmacokinetic_regimen/`](../../math_explorer_gui/src/tabs/medical/pharmacokinetic_regimen/mod.rs).
   - Enforce `URL_JSON_BOUNDS` (`4,096` decoded bytes) with compact `CompactDoseEvent` serialization so schedules up to `64` events round-trip losslessly via `#tab=medical&tool=PharmacokineticRegimenTool&state=...`.
   - Declare `ExportCapabilities::plot()` with offscreen `ExportPrimitive` therapeutic-window polygons for SVG/PNG/CSV/JSON exports, and integrate [`math_explorer_gui/src/web_io.rs`](../../math_explorer_gui/src/web_io.rs) `spawn_save_as_file_dialog` downloads for CSV dosing schedules and JSON exposure reports.
4. **Traceability Alignment**:
   - Point `pharmacokinetics` in [`traceability.toml`](../../traceability.toml) and module `[cite:pharmacokinetics]` annotations to `clinical_trials_statistics.tex`.

## Consequences
- **Positive**: Variable-dose schedules accurately model loading doses (`66.49%` to `86.89%` Time-in-Therapeutic-Range) and fractional catch-up regimens (eliminating the `29.76%` peak underestimation of uniform `SuperpositionModel` while distinguishing safe `1.5D` recovery from `8.32%` toxic `2D` overshoot) with zero heap allocations in verified regions.
- **Positive**: Client-side native and `wasm32-unknown-unknown` builds remain 100% network-free, respecting `'self'` CSP and URL payload bounds.
- **Neutral**: Interactive schedules are bounded to `MAX_REGIMEN_EVENTS = 64` administrations; nonlinear clearance and multi-compartment ODEs remain out of scope for analytical Bateman superposition.
