# ADR 20261123: Bridge AccessKit to Browser DOM in `eframe::WebRunner` and Opt-In Speech Controls

## Status
Accepted (extends [ADR 0032](0032-high-contrast-focus-selection-and-raster-tokens.md), [ADR 0033](0033-wasm32-browser-tests-and-playwright-ci-smoke-gate.md), and [ADR 20261123](20261123-tune-dev-profiles-and-inline-atomic-metrics.md))

## Context
1. **Unforwarded `AccessKit` Semantic Trees on `wasm32`**:
   - Upstream `eframe` `0.33.3` omits `egui_ctx.enable_accesskit()` in `AppRunner::new` on `wasm32` and drops `PlatformOutput::accesskit_update` inside `AppRunner::handle_platform_output`, leaving `accesskit_node_builder` annotations (`Role::Region`, `Role::Splitter`, `Role::Button`, `Role::GraphicsDocument`, `Role::Image`) invisible to browser assistive technology while `<canvas id="the_canvas_id" role="application">` intercepted screen-reader browse mode.
2. **Unprompted `window.speechSynthesis` Narration and Duplicate Boot Announcements**:
   - Enabling `eframe`'s `web_screen_reader` feature by default combined with unconditional `speak_text` calls in `accessibility/hover.rs` caused unprompted speech synthesis on focus/hover without independent volume or cancel controls, while `#aria-live-polite` had no synchronized single-shot guard across Rust `finish_wasm_startup` and JavaScript `markBootComplete`.
3. **`wasm32` Panic-Abort Test Gate in `verified_engine`**:
   - `metrics::tests::test_run_verified_panic_unwind_preserves_atomic_count` in [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) invokes `std::panic::catch_unwind`, which traps with `unreachable` on `wasm32-unknown-unknown` (`panic = "abort"`).

## Decision
1. **Patched `eframe` `0.33.3` Web AccessKit Bridge (`patches/eframe` & `math_explorer_gui/src/accessibility/`)**:
   - Patch `eframe` `0.33.3` via `[patch.crates-io]` in root [`Cargo.toml`](../../Cargo.toml) (`exclude = ["patches/eframe"]`), removing `web_screen_reader` from default features, enabling `egui_ctx.enable_accesskit()` in [`patches/eframe/src/web/app_runner.rs`](../../patches/eframe/src/web/app_runner.rs), forwarding `WebAccessKitOutput` and `destroy()` hooks through [`patches/eframe/src/web/web_runner.rs`](../../patches/eframe/src/web/web_runner.rs), and preserving DOM focus inside `#accesskit_dom_root`.
   - Reconcile `accesskit::TreeUpdate` into semantic HTML/ARIA nodes inside `<div id="accesskit_dom_root">` beside `<canvas id="the_canvas_id">` (with `role="application"` removed) in [`math_explorer_gui/index.html`](../../math_explorer_gui/index.html), routing `focus` (capture), `click`, `input`, and `keydown` DOM events back into `egui::Event::AccessKitActionRequest` on both initial boot ([`math_explorer_gui/src/main.rs`](../../math_explorer_gui/src/main.rs)) and WebGL context recovery ([`math_explorer_gui/src/webgl_recovery.rs`](../../math_explorer_gui/src/webgl_recovery.rs)).
2. **Opt-In Speech Narration and Synchronized Polite Ready Announcement**:
   - Default `o.screen_reader` to `false` on `wasm32` and gate `speak_text` in [`math_explorer_gui/src/accessibility/speech.rs`](../../math_explorer_gui/src/accessibility/speech.rs) behind persisted `speech_narration_enabled` and `speech_narration_volume` fields in [`math_explorer_gui/src/app/state.rs`](../../math_explorer_gui/src/app/state.rs), exposed with a `Stop speech` (`speechSynthesis.cancel()`) button in [`math_explorer_gui/src/app/ui_panels.rs`](../../math_explorer_gui/src/app/ui_panels.rs) while keeping theory descriptions silent until opted in.
   - Coordinate `announce_startup_ready_once` in [`math_explorer_gui/src/accessibility/live_region.rs`](../../math_explorer_gui/src/accessibility/live_region.rs) with [`math_explorer_gui/reload.js`](../../math_explorer_gui/reload.js) and [`math_explorer_gui/initializer.js`](../../math_explorer_gui/initializer.js) via `data-startup-ready-announced` and twin-slot `#aria-live-polite-slot-{0,1}` alternation so `"Math Explorer ready. Interactive simulation workspace loaded."` is announced once per boot/recovery cycle under strict CSP (`script-src 'self' 'wasm-unsafe-eval' blob:`).
3. **Host-Only Unwind Test Gate and Vendor Patch Report Exclusion (`crates/verified_engine/src/metrics.rs`, `crates/unified_verification/src/report.rs`)**:
   - Gate `test_run_verified_panic_unwind_preserves_atomic_count` with `#[cfg(not(target_arch = "wasm32"))]` alongside `test_multithreaded_active_verifiers_isolation` so `xtask test-wasm` executes cleanly on `wasm32-unknown-unknown`, and exclude `patches/` in [`crates/unified_verification/src/report.rs`](../../crates/unified_verification/src/report.rs) so first-party `wasm_path_coverage_pct` and assertion density metrics are unaffected by the vendored `eframe` patch crate.

## Verification Commands
```bash
cargo test -p math_explorer_gui --all-features -- accessibility markup canvas_accessibility
cargo run -p xtask -- test-wasm --screenshot-dir /tmp/smoke_screenshots
cargo run -p xtask -- lint-all
```

## Consequences
- **Positive**: Browser assistive technology receives live semantic landmarks, headings, splitters, buttons, sliders, and plot/canvas nodes inside `#accesskit_dom_root` with bidirectional keyboard/focus/click routing across initial startup and WebGL context recovery.
- **Positive**: `window.speechSynthesis` remains completely silent until explicitly enabled by the user, with persisted independent volume and immediate `cancel()` control, while WASM unit tests and Playwright accessibility smoke gates pass end to end.
- **Neutral**: Maintaining `patches/eframe` requires re-evaluating upstream `eframe` web `AccessKit` support when upgrading `egui`/`eframe` past `0.33.3`.
