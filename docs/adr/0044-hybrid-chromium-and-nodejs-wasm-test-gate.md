# ADR 0044: Hybrid Headless Chromium and Node.js WebAssembly Test Gate

## Status

Accepted

## Context

`OxidizeMath` compiles and tests two distinct categories of `wasm32-unknown-unknown` targets:

1. **Browser-bound crates** (`oxidize_core`, `egui_plot`, and `math_explorer_gui`) that link `web-sys` / `js-sys` DOM APIs and register `#[wasm_bindgen_test]` suites executed through `WasmBindgenTestContext` under the production Content Security Policy (`PRODUCTION_CSP`).
2. **Pure-Rust scientific and engine libraries** (the 12 non-proc-macro crates under `crates/*` plus the `math_explorer` facade) whose `#[test]` functions use standard `libtest` (`exports.main(0, 0)`) without browser DOM bindings.

Previously:

- Launching Playwright and headless Chromium for every pure-Rust `libtest` binary added unnecessary browser startup overhead, while binaries importing `wasm-bindgen` bindings without `WasmBindgenTestContext` or runnable `__wbgt_*` symbols risked silently exiting `0` when 0 tests matched.
- `math_explorer_gui/src/web_io.rs` still carried a `#[cfg(target_arch = "wasm32")]` `#[test]` function (`test_spawn_save_direct_file_wasm_typecheck_and_channel_delivery`), which lived in the `libtest` pool rather than the `#[wasm_bindgen_test]` browser harness, and static verification did not reject new `#[cfg(target_arch = "wasm32")]` `#[test]` items.
- `xtask test-wasm` invoked Cargo's implicit test runner instead of building `.wasm` test executables with `--no-run --message-format=json` and invoking Python directly via `Command` argument vectors, and `.github/workflows/ci.yml` ran redundant `cargo check --workspace --target wasm32-unknown-unknown` passes immediately before `cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings`.

## Decision

1. **Dual Execution Modes & Fail-Closed Contract in [scripts/wasm_test_runner.py](../../scripts/wasm_test_runner.py)**:
   - Binaries exporting `WasmBindgenTestContext` (`__wbg_wasmbindgentestcontext_new`/`run`) and `__wbgt_*` tests run via `wasm-bindgen --target web` in Playwright headless Chromium served with `PRODUCTION_CSP` headers.
   - Pure-Rust `libtest` binaries (zero non-inert host/`wasm-bindgen` imports and an exported `main(argc, argv)`) run directly in Node.js (`node`) via `instance.exports.main(0, 0)`, patching `TestDesc.ignore` offsets in WASM linear memory so `--skip`, `--exact`, `--ignored`, `--include-ignored`, and positional filters are honored at runtime.
   - Any binary importing non-inert `wasm-bindgen` or host bindings without `WasmBindgenTestContext` and runnable `__wbgt_*` tests (or missing `class WasmBindgenTestContext` in generated JS glue) fails closed with a non-zero exit code.
2. **Static Rejection of Orphaned `wasm32` `#[test]` Items**:
   - Converted `test_spawn_save_direct_file_wasm_typecheck_and_channel_delivery` in [math_explorer_gui/src/web_io.rs](../../math_explorer_gui/src/web_io.rs) to `#[wasm_bindgen_test]`.
   - Extended [crates/unified_verification/src/wasm_coverage.rs](../../crates/unified_verification/src/wasm_coverage.rs), [crates/unified_verification/src/report.rs](../../crates/unified_verification/src/report.rs), and [crates/unified_verification/src/policy_audit.rs](../../crates/unified_verification/src/policy_audit.rs) to audit `egui_plot` and reject any `#[cfg(target_arch = "wasm32")]` `#[test]` function across the workspace.
3. **JSON Artifact Extraction & Hybrid `xtask test-wasm` Pipeline**:
   - Extended [crates/oxidize_core/src/workspace_wasm.rs](../../crates/oxidize_core/src/workspace_wasm.rs) and [apps/xtask/src/wasm_gate.rs](../../apps/xtask/src/wasm_gate.rs) to classify `BrowserChromium` (`oxidize_core`, `egui_plot`, `math_explorer_gui`) and `NodeJs` (12 pure-Rust `crates/*` libraries + `math_explorer`) targets, compile test artifacts with `--no-run --message-format=json`, parse `.wasm` executable paths from Cargo JSON output, and execute each artifact via `Command` `OsStr` argv without space-splitting.
   - Removed duplicate `wasm32` `cargo check` steps from [.github/workflows/ci.yml](../../.github/workflows/ci.yml) and [AGENTS.md](../../AGENTS.md), and added `node` probing to `xtask setup`.

## Consequences

- Browser `#[wasm_bindgen_test]` suites continue to run under real Chromium DOM/WebGL2 + `PRODUCTION_CSP` enforcement while all 13 pure-Rust workspace libraries execute fast deterministic `wasm32` `libtest` runs in Node.js.
- Misconfigured `wasm-bindgen` binaries without `WasmBindgenTestContext` and orphaned `#[cfg(target_arch = "wasm32")]` `#[test]` attributes fail closed in both runtime and static gates.
- `cargo test --target wasm32-unknown-unknown` from crate subdirectories and workspace paths containing spaces works seamlessly via `.cargo/config.toml` (`CARGO_MANIFEST_DIR`) and `xtask`'s JSON artifact runner.
