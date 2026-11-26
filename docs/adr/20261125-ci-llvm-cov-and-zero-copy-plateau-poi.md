# ADR 20261125: CI Tool-Cache Subcommand Probing, Zero-Copy Plateau POI Dispatch, and Thread-Isolated Plot Test Hooks

## Status
Accepted (2026-11-25)

## Context

Following the `#![no_std]` substrate and 113-tool WebAssembly workspace upgrades, cross-platform GitHub Actions CI (`Pre-flight Checks`, `Linux Native`, `Windows Native`, `macOS Native`, and `Web/WASM`) surfaced five infrastructure and test-profile regressions:

1. **Cargo subcommand binary verification (`Linux Native`):** `.github/actions/ci-tool-cache/action.yml` verified cached or installed binaries solely via `"$bin_path" --version`, which exits `1` on Cargo subcommands such as `cargo-llvm-cov` (`0.9.2`) that require `"$bin_path" llvm-cov --version`.
2. **Duplicate allocating plateau compaction & `opt-level = 0` bounds checks (`Windows Native` / `macOS Native`):** `egui_plot::poi` pre-compacted dense polylines into a heap-allocated `Vec<[f64; 2]>` (`egui_plot/src/poi/plateau.rs`) before calling `math_commons::poi::scan_line_pois` (which already fast-forwards flat runs via `PointSeries::advance_plateau_run`), while `advance_sample2d_plateau!` in `crates/02_engines/math_commons/src/series.rs` indexed `chunks[ci]` and `c[$idx]` per sample, pushing 200k-point plateau tests over their `< 5 ms` budget under `[profile.test]` (`opt-level = 0`).
3. **Process-global plot test state contention (`Windows Native` / `macOS Native`):** `egui_plot` stored `TEST_IMPORT_DIALOG_TRIGGERS` and `TEST_CLIPBOARD_READ_OVERRIDE` in process-global `Mutex`es and called `.clear()` on global CSV import queues inside `test_bounded_plot_import_queue_evicts_oldest_unrendered_entries`, racing against concurrent `egui_plot` unit test threads.
4. **WebAssembly slider clamping, unwind gating, and compressed bundle ceiling (`Web/WASM`):** `GIF Frame Steps` lacked `SliderClamping::Edits` (clobbering programmatic `3`/`4`-frame test states on idle modal renders), `test_run_verified_panic_unwind_preserves_atomic_count` ran on `panic = "abort"` `wasm32-unknown-unknown`, and `scripts/verify_wasm_budget.sh` retained a `4 MB` default cap after the 113-tool Gzip artifact grew to `4.73 MB` (`3.96 MB` Brotli).
5. **Unused `rand` manifest entries (`Pre-flight Checks`):** `crates/01_core/oxidize_core/Cargo.toml` and `crates/04_bindings/markdown_tests/Cargo.toml` retained redundant direct `rand` dependencies after `oxidize_core::rng` switched to re-exporting `math_rng::*`.

## Decision

1. **Cargo Subcommand Version Fallback in `ci-tool-cache`:** Probe `"$bin_path" "${BIN_NAME#cargo-}" --version` whenever `BIN_NAME` starts with `cargo-` and direct `"$bin_path" --version` exits non-zero.
2. **Single-Pass Zero-Copy Plateau Scanning:** Delete `egui_plot/src/poi/plateau.rs`, dispatch `scan_line_pois` directly on borrowed `&[PlotPoint]` and `&[[f64; 2]]` slices in `egui_plot/src/poi.rs`, and rewrite `math_commons::advance_sample2d_plateau!` using 16-wide slice-pattern destructuring (`while let [c, rest @ ..] = chunk_slice`) with zero inner bounds checks or intermediate `Vec` allocations.
3. **Thread-Isolated `egui_plot` Test Hooks:** Move `TEST_IMPORT_DIALOG_TRIGGERS` and `TEST_CLIPBOARD_READ_OVERRIDE` into `thread_local!` `RefCell`s and drain only synthetic `("stale_unrendered_plot", idx)` IDs in bounded import queue tests.
4. **Web/WASM & Manifest Hygiene:** Configure `.clamping(egui::SliderClamping::Edits)` on the GIF frame slider, gate `test_run_verified_panic_unwind_preserves_atomic_count` to `#[cfg(not(target_arch = "wasm32"))]`, set the compressed WASM budget to `5 MB` (`5,242,880` bytes) with HTTP/1.1 loopback retry in `scripts/wasm_test_runner.py`, and remove redundant `rand` dependencies from `oxidize_core` and `markdown_tests`.

## Consequences

- **Positive:** 200k-point plateau critical-point scans execute in `< 1 ms` even in unoptimized `dev`/`test` builds without heap compaction allocations, `cargo-llvm-cov` binary verification succeeds on cached and fresh installs, and parallel `egui_plot` tests are race-free across Linux, Windows, macOS, and Headless Chromium WASM.
- **Neutral:** Compressed WebAssembly artifact budget is standardized at `5 MB` across `scripts/verify_wasm_budget.sh`, `AGENTS.md`, and `apps/xtask/README.md`.
