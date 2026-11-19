# ADR 0022: Enforce WebAssembly Clippy Guardrails in Pre-Commit Hooks and CI

## Status
Accepted

## Context
Both `cargo run -p xtask -- lint-all` (invoked by `.githooks/pre-commit`) and `.github/workflows/ci.yml` previously ran `cargo clippy -- -D warnings` exclusively on native host targets, while the `Web/WASM` (`wasm32-unknown-unknown`) CI job executed `cargo check` and `trunk build` without Clippy. Because `#[cfg(target_arch = "wasm32")]` modules are skipped during host compilation, WebAssembly-only lint violations (`clippy::io_other_error` in `crates/oxidize_core/src/vfs.rs` and `unused_mut` in `math_explorer_gui/src/tabs/traceability.rs`) merged onto `main` undetected.

## Decision
1. Resolved WebAssembly Clippy lint errors across workspace crates without weakening `[workspace.lints]` or `.clippy.toml` thresholds:
   - Replaced five `std::io::Error::new(std::io::ErrorKind::Other, ...)` calls in `WasmVfs::read_to_string` (`crates/oxidize_core/src/vfs.rs`) with `std::io::Error::other(...)`.
   - Gated the desktop-only `new_path_picked` binding and folder-selection branch behind `#[cfg(not(target_arch = "wasm32"))]` in `math_explorer_gui/src/tabs/traceability.rs`.
2. Consolidated markdown link helpers in `apps/xtask/src/markdown_links.rs` (preserving the 500-line file limit) and extracted `run_clippy_lints` from `lint_all` in `apps/xtask/src/main.rs` (preserving the 60-line function limit) to run strict WebAssembly Clippy (`--workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings`, omitting `--benches` so host-only Criterion benchmarks are skipped).
3. Added a matching `Run Clippy (WASM)` step to the `build-and-test` job in `.github/workflows/ci.yml` and documented local WebAssembly Clippy commands in `AGENTS.md` and `CONTRIBUTING.md`.

## Consequences
- Both `.githooks/pre-commit` (`xtask lint-all`) and the `Web/WASM` CI matrix leg reject `wasm32-unknown-unknown` Clippy warnings before merge.
- All public Rust APIs, native Clippy steps, WebAssembly `cargo check` / `trunk build` steps, and workspace lint thresholds remain unchanged.
- Satisfies the `verify-records` architectural record requirement for `crates/oxidize_core` updates.
