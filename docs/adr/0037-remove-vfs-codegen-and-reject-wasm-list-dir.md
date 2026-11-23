# ADR 0037: Remove `oxidize_core` VFS Build-Script Codegen and Reject WebAssembly Directory Listing

## Status
Accepted (extends [ADR 0027](0027-build-rerun-directives-and-cargo-cache.md) and [ADR 0032](0032-precomputed-traceability-manifest-and-wasm-vfs-subpaths.md))

## Context
1. **Unreferenced `WasmVfs::list_dir` Table Generation**: After [ADR 0032](0032-precomputed-traceability-manifest-and-wasm-vfs-subpaths.md) replaced live browser workspace scanning with precomputed `./traceability_manifest.json` fetches in [`math_explorer_gui/src/tabs/traceability/web_snapshot.rs`](../../math_explorer_gui/src/tabs/traceability/web_snapshot.rs) and gated live `VirtualFileSystem` scanning in [`math_explorer_gui/src/tabs/traceability/scanner.rs`](../../math_explorer_gui/src/tabs/traceability/scanner.rs) to non-`wasm32` targets, `WasmVfs::list_dir` in [`crates/oxidize_core/src/vfs.rs`](../../crates/oxidize_core/src/vfs.rs) had zero production callers. Retaining dead codegen conflicted with [ADR 0019](0019-purge-dead-code-bypasses.md).
2. **False Cross-Crate Cargo Cache Invalidations**: `crates/oxidize_core/build.rs` still walked every directory across `../../papers`, `../../math_explorer/src`, and `../../math_explorer_gui/src/tabs`, emitting `cargo:rerun-if-changed` directives solely for that unused static directory table. Editing a LaTeX manuscript or adding a GUI tab directory invalidated cached artifacts for `oxidize_core` and all downstream crates that depend on it.

## Decision
1. **Delete `crates/oxidize_core/build.rs` and Dead Build-Output Tests**:
   - Remove `crates/oxidize_core/build.rs` and its `OUT_DIR` `include!` wiring and host rerun-directive unit test in [`crates/oxidize_core/src/vfs.rs`](../../crates/oxidize_core/src/vfs.rs).
   - Remove the `oxidize_core` build-output rerun assertions from [`crates/unified_verification/src/workflow_linter/cargo_cache_tests.rs`](../../crates/unified_verification/src/workflow_linter/cargo_cache_tests.rs) while preserving the `generated_tabs.rs` checks for `math_explorer_gui`.
2. **Return `ErrorKind::Unsupported` from `WasmVfs::list_dir` Without API Drift**:
   - Update `WasmVfs::list_dir` in [`crates/oxidize_core/src/vfs.rs`](../../crates/oxidize_core/src/vfs.rs) to return `std::io::ErrorKind::Unsupported` while preserving `VirtualFileSystem`, `pub type DefaultVfs = WasmVfs`, `resolve_wasm_fetch_url`, `WasmVfs::read_to_string`, `WasmVfs::write_to_file`, and `trigger_download`, leaving [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) and [`crates/oxidize_core/src/path_utils.rs`](../../crates/oxidize_core/src/path_utils.rs) unchanged.
3. **Update Browser Unit Tests**:
   - Replace the embedded `papers` directory assertion in [`crates/oxidize_core/src/vfs_wasm_tests.rs`](../../crates/oxidize_core/src/vfs_wasm_tests.rs) with a `#[wasm_bindgen_test]` verifying `ErrorKind::Unsupported` for `WasmVfs::list_dir` and `DefaultVfs::list_dir`, while retaining HTTP `fetch` (`200` and `404`) and `Blob` download browser tests per [ADR 0033](0033-wasm32-browser-tests-and-playwright-ci-smoke-gate.md).

## Consequences
- **Positive**: Editing files or directories under `papers/`, `math_explorer/src`, or `math_explorer_gui/src/tabs` no longer triggers `oxidize_core` rebuilds or downstream crate cache invalidations.
- **Positive**: Eliminates generated static directory tables from `wasm32` builds without introducing dead-code suppressions or whitelist entries.
- **Neutral**: Public `VirtualFileSystem`, `DefaultVfs`, and `WasmVfs` signatures in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) and host directory listing (`DefaultVfs::list_dir` and `RootedVfs::list_dir`) remain unchanged.
