# ADR 20261123: Coupled Electrochemical Battery Aging, Zero-Allocation CSV Calibration, and Multi-Segment Lifetime Exports

## Status

Accepted

## Context

`domain_applied::applied::battery_degradation` and the `math_explorer_gui` battery calculators previously modeled only an isothermal two-parameter power law (`PowerLawModel::standard()` with $N_{70}(D) = \alpha D^\beta$ and $C(N) = 0.7^{N / N_{70}}$):

1. `CapacityFadeTool` in `math_explorer_gui/src/tabs/battery_degradation/capacity_fade.rs` rendered `Temperature (°C)` inside `ui.add_enabled_ui(false, ...)` and ignored stretched cycling exponent $z$, charge/discharge C-rate stress, Arrhenius thermal activation $\exp(-E_a / (R T))$, and $\sqrt{t}$ solid-electrolyte-interphase (SEI) calendar growth. At $45\,^\circ\text{C}$ ($318.15\,\text{K}$), isothermal projections overestimated NMC811 cycle life by `122.29%` and mispredicted $80\%$ first-life crossings for diffusion-limited SEI exponents ($z \approx 0.50$).
2. `LifetimeEstimatorTool` in `math_explorer_gui/src/tabs/battery_degradation/lifetime_estimator.rs` returned `ExportCapabilities::none()`, accepted only static scalar inputs (`dod`, `target_capacity`, `cycles_per_day`), rendered no state-of-health trajectory plot, and could not export calibrated cell specifications or yearly warranty schedules.
3. `traceability.toml` mapped `battery_degradation` to `algorithmic_information_rust.tex`, while `capacity_fade.rs` cited `mmwave_radiotherapy_setup`.

At the same time, existing public callers and `crates/unified_verification/api_baseline.json` depend on `PowerLawModel` and `DegradationModel` remaining intact, `VerifiedAllocator` requires zero dynamic heap allocations inside `#[verified_engine::verified]` kernels, and `URL_JSON_BOUNDS` caps decoded `#state=` URL payloads at `4,096` bytes.

## Decision

1. **Coupled Electrochemical Aging Kernel and Presets in `domain_applied`**:
   - Retain `PowerLawModel` and `DegradationModel` unchanged in [`crates/domain_applied/src/applied/battery_degradation/model.rs`](../../crates/03_domains/domain_applied/src/applied/battery_degradation/model.rs) and add `ChemistryPreset::{Nmc811, Lfp, Nca, Lto}` plus `ElectrochemicalAgingModel`, re-exporting them from [`crates/domain_applied/src/applied/battery_degradation/mod.rs`](../../crates/03_domains/domain_applied/src/applied/battery_degradation/mod.rs).
   - Couple DoD power-law stress $D^\beta$, stretched cycling exponent $z \in [0.1, 2.0]$, Arrhenius thermal acceleration $\exp((E_a / R)(1 / T_{\text{ref}} - 1 / T))$ normalized at $25\,^\circ\text{C}$ across $[-20.0, 60.0]\,^\circ\text{C}$ (`253.15..=333.15 K`), C-rate mechanical stress $I_c^{\gamma_c}$ across $[0.05, 10.0]$, and parabolic $\sqrt{t}$ SEI calendar fade under `#[verified_engine::verified]`.
2. **Stack-Allocated Levenberg-Marquardt Calibration and Multi-Segment Duty Solver**:
   - Add [`crates/domain_applied/src/applied/battery_degradation/calibration.rs`](../../crates/03_domains/domain_applied/src/applied/battery_degradation/calibration.rs) invoking `math_commons::fit::levenberg_marquardt_numeric` over smooth $[\ln N_{70,\text{eff}}, \operatorname{logit}(z)]$ coordinates with zero heap allocations.
   - Add [`crates/domain_applied/src/applied/battery_degradation/duty.rs`](../../crates/03_domains/domain_applied/src/applied/battery_degradation/duty.rs) (`DutySegment`, `evaluate_duty_point`, `estimate_duty_lifetime`) supporting `1..=8` duty phases and integrating strictly the $\sqrt{t}$ SEI calendar term when `daily_cycles == 0.0` (storage) without division by zero.
3. **Interactive CSV Calibration, Duty Mission Builder, and Client-Side Exports in `math_explorer_gui`**:
   - Split [`capacity_fade.rs`](../../apps/math_explorer_gui/src/tabs/battery_degradation/capacity_fade.rs) and [`lifetime_estimator.rs`](../../apps/math_explorer_gui/src/tabs/battery_degradation/lifetime_estimator.rs) into modular subtrees (`<= 500` lines per file, `<= 60` lines per function) backed by [`cell_card.rs`](../../apps/math_explorer_gui/src/tabs/battery_degradation/cell_card.rs) and [`web_io.rs`](../../apps/math_explorer_gui/src/web_io.rs) (`spawn_open_file_dialog`, `spawn_save_as_file_dialog`, `strip_utf8_bom`).
   - Enable `Temperature (°C)`, `ChemistryPreset`, `C-Rate`, and `Stretched Exponent (z)` controls in `CapacityFadeTool`, cap imported `.csv` logs at `2,000` finite rows with rollback-on-error semantics, render `egui_plot` SOH residuals, and export `.json` cell cards.
   - Upgrade `LifetimeEstimatorTool` to `ExportCapabilities::plot()` (updating [`deferred_export_audit_tests.rs`](../../apps/math_explorer_gui/src/framework/deferred_export_audit_tests.rs)), add a `1..=8` commute/fast-charge/storage segment builder fitting within `URL_JSON_BOUNDS`, plot coupled SOH trajectories, and export `.json` cell cards plus formula-safe yearly `.csv` warranty schedules (`80%` and `70%` milestones).
4. **Traceability and Public API Baseline Alignment**:
   - Align `// [cite:battery_degradation]` across `domain_applied`, `math_explorer`, and `math_explorer_gui` and map `battery_degradation = "self_calibration_paper.tex"` in [`traceability.toml`](../../traceability.toml).
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../apps/unified_verification/api_baseline.json) via `cargo run -p xtask -- regenerate-baseline`.

## Consequences

- **Positive**: Eliminates thermal and stretched-exponent cycle-life skew while preserving 100% client-side Rust/WebAssembly execution, zero new crate dependencies, zero allocations inside `VerifiedEngine::run_verified`, and backward-compatible `#state=` URL / starter preset serialization via `#[serde(default, skip_serializing_if = ...)]`.
- **Trade-off**: Expanding `domain_applied::applied::battery_degradation` and `math_explorer::applied::battery_degradation` public exports requires updating `api_baseline.json` alongside `deferred_export_audit_tests.rs` capability counts.
