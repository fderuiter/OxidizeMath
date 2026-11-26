# ADR 20261125: 3D Isometric Orientation Gizmos, Accessibility Sidecars, and WASM CI Hygiene

## Status
Accepted (2026-11-25)

## Context

1. **3D Viewport Orientation & Keyboard/Screen-Reader Parity:** Interactive 3D workbenches across `egui_plot` and `math_explorer_gui` lacked a unified isometric orientation gizmo, keyboard camera presets, and structured AccessKit sidecar readouts exposing live azimuth, elevation, zoom, and world-space bounds.
2. **Post-`math_rng` Unused `rand` Dependencies:** Following the extraction of `OxidizeRng` into [`crates/00_foundation/math_rng`](../../crates/00_foundation/math_rng), [`crates/oxidize_core/Cargo.toml`](../../crates/oxidize_core/Cargo.toml) and [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml) retained unused `rand` dependencies after updating [`crates/oxidize_core/README.md`](../../crates/oxidize_core/README.md) doctests to import `OxidizeRng` directly, tripping both `cargo-machete` and `unified_verification` policy audits.
3. **WebAssembly Test & Release Budget Constraints:**
   - [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) invoked `std::panic::catch_unwind` inside `test_run_verified_panic_unwind_preserves_atomic_count`, trapping with `RuntimeError: unreachable` on `wasm32-unknown-unknown` (`panic = "abort"`).
   - [`egui_plot/src/export_modal.rs`](../../egui_plot/src/export_modal.rs) left the GIF frame-count `egui::Slider` (`5..=60`) on `SliderClamping::Always`, overriding programmatic sub-minimum test frame counts (`4`) during incremental WASM GIF export ticks.
   - Adding 3D orientation gizmos, mesh exporters, and accessibility sidecars brought the gzipped release `.wasm` bundle to ~4.74 MB, exceeding the legacy 4 MB cap in [`scripts/verify_wasm_budget.sh`](../../scripts/verify_wasm_budget.sh).
4. **Cargo Subcommand Tool Cache Verification:** [`.github/actions/ci-tool-cache/action.yml`](../../.github/actions/ci-tool-cache/action.yml) verified cached/installed binaries exclusively via `"$bin_path" --version`, causing pure Cargo subcommand binaries such as `cargo-llvm-cov` (`cargo-llvm-cov llvm-cov --version`) to exit with status `2`.

## Decision

1. **3D Orientation Gizmos & Accessibility Sidecars:** Provide `Sidecar3d` in [`egui_plot/src/sidecar_3d.rs`](../../egui_plot/src/sidecar_3d.rs) and `CameraGizmo` + keyboard/AccessKit camera announcements in [`math_explorer_gui/src/camera_gizmo.rs`](../../math_explorer_gui/src/camera_gizmo.rs) and [`math_explorer_gui/src/camera/mod.rs`](../../math_explorer_gui/src/camera/mod.rs) across 3D simulation viewports.
2. **Dependency & Doctest Cleanup:** Remove unused `rand` entries from [`crates/oxidize_core/Cargo.toml`](../../crates/oxidize_core/Cargo.toml) and [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml) and import `oxidize_core::rng::{OxidizeRng, RngCore}` directly in [`crates/oxidize_core/README.md`](../../crates/oxidize_core/README.md).
3. **WASM Test Gating, Slider Clamping, and 5 MB Budget:**
   - Gate `test_run_verified_panic_unwind_preserves_atomic_count` in [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) with `#[cfg(not(target_arch = "wasm32"))]`.
   - Set `.clamping(egui::SliderClamping::Edits)` on the GIF frame-count slider in [`egui_plot/src/export_modal.rs`](../../egui_plot/src/export_modal.rs) so interactive edits remain bounded to `5..=60` without mutating programmatic test configurations.
   - Raise the default compressed `.wasm.gz` release budget in [`scripts/verify_wasm_budget.sh`](../../scripts/verify_wasm_budget.sh), [`AGENTS.md`](../../AGENTS.md), and [`apps/xtask/README.md`](../../apps/xtask/README.md) from `4 MB` (`4,194,304` bytes) to `5 MB` (`5,242,880` bytes).
4. **Cargo Subcommand Verification in `ci-tool-cache`:** Probe `"$bin_path" "${BIN_NAME#cargo-}" --version` with fallback to `"$bin_path" --version` for `cargo-*` binaries in [`.github/actions/ci-tool-cache/action.yml`](../../.github/actions/ci-tool-cache/action.yml).

## Consequences

- **Positive:** 3D viewports expose consistent isometric camera controls and screen-reader sidecar semantics while native, cross-target, and `wasm32-unknown-unknown` CI suites pass with zero unused dependencies or subcommand verification failures.
- **Neutral:** Compressed `.wasm.gz` release ceiling increases from `4 MB` to `5 MB` (~260 KiB headroom above the ~4.74 MB bundle).
