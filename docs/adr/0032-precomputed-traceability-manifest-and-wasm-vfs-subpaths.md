# ADR 0032: Precomputed Traceability Manifest, Subpath-Safe `WasmVfs`, and Strict Scan Error Propagation

## Status
Accepted (extends [ADR 0003](0003-traceability-refactoring.md), [ADR 0022](0022-wasm-clippy-guardrails.md), and [ADR 0027](0027-build-rerun-directives-and-cargo-cache.md))

## Context
Three architectural gaps degraded the static WebAssembly runtime on GitHub Pages (`/OxidizeMath/`) and repository traceability diagnostics:
1. **Root-Escaped `WasmVfs` Fetch Paths**: In [`crates/oxidize_core/src/vfs.rs`](../../crates/oxidize_core/src/vfs.rs), `WasmVfs::read_to_string` prefixed relative paths with `"/"` (`format!("/{}", url)`), escaping subpath deployments such as `/OxidizeMath/` to the domain root (`/`). Meanwhile, `DefaultVfs::list_dir` swallowed `std::fs::read_dir` failures instead of returning `Err`.
2. **Silent Read Failures and Non-Serializable `TraceabilityReport`**: In [`crates/oxidize_core/src/traceability.rs`](../../crates/oxidize_core/src/traceability.rs), `TraceabilityReport` lacked `serde::Serialize` and `serde::Deserialize` derives, `verify_module_registered` stubbed `true` on `wasm32`, and `TraceabilityEngine::scan_repository` swallowed `NotFound` and I/O read failures across `papers/`, [`traceability.toml`](../../traceability.toml), and discovered `.rs` files—returning `Ok(report)` with 28 false `orphaned_papers` rather than propagating an explicit error.
3. **Web Manifest Contract and Hosted PDF Links**: Static web builds in [`math_explorer_gui/build_snapshot.rs`](../../math_explorer_gui/build_snapshot.rs) and [`math_explorer_gui/src/tabs/traceability.rs`](../../math_explorer_gui/src/tabs/traceability.rs) needed to serve a single precomputed `./traceability_manifest.json` matching `TraceabilityReport` across all 28 academic papers and workspace crates, while resolving relative `./{stem}.pdf` manuscript links against PDFs staged from `compile-papers` into GitHub Pages.

## Decision
1. **Subpath-Relative URL Resolution in `WasmVfs`**:
   - Prefix relative `WasmVfs::read_to_string` paths (those not starting with `http://`, `https://`, `/`, or `./`) with `"./"` via `resolve_wasm_fetch_url` so browser `fetch` calls resolve within `/OxidizeMath/` instead of escaping to `/`.
   - Propagate `std::fs::read_dir` errors from `DefaultVfs::list_dir` consistently with `WasmVfs::list_dir`.
2. **Serializable `TraceabilityReport` and Strict `scan_repository` Error Propagation**:
   - Derive `serde::Serialize` and `serde::Deserialize` (`#[serde(default)]`) on `TraceabilityReport` in [`crates/oxidize_core/src/traceability.rs`](../../crates/oxidize_core/src/traceability.rs) and extract citation/AST analysis helpers into [`crates/oxidize_core/src/traceability/analysis.rs`](../../crates/oxidize_core/src/traceability/analysis.rs) to stay well below the 500-line ceiling.
   - Validate `verify_module_registered` on `wasm32` against embedded [`traceability.toml`](../../traceability.toml) metadata instead of stubbing `true`.
   - Propagate `std::io::Error` from `scan_papers_dir`, `read_registry_table`, and `process_single_file` in `TraceabilityEngine::scan_repository` so unreadable manifests, missing paper directories, or failed source reads transition `TraceabilityTab` to `TraceabilityScanState::Failed` with a `Retry` action instead of reporting false orphaned papers.
3. **Precomputed `traceability_manifest.json` and Staged PDF Hyperlinks**:
   - Generate `traceability_manifest.json` during `math_explorer_gui` compilation, stage and pre-compress (`.gz`, `.br`) it in [`scripts/finalize_web_dist.sh`](../../scripts/finalize_web_dist.sh), and load `./traceability_manifest.json` lazily via `WasmVfs` when `TraceabilityTab` renders or refreshes.
   - Emit `./{stem}.pdf` links on `wasm32` and `https://fderuiter.github.io/OxidizeMath/{stem}.pdf` links on desktop in `TraceabilityTab::paper_url`, backed by compiled manuscript PDFs staged into `pages_dist/` in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

## Consequences
- **Positive**: Cold boot under `/OxidizeMath/` performs zero traceability requests and emits zero HTTP `404` or console errors; opening `Traceability Portal` fetches `./traceability_manifest.json` once with full 28-paper workspace crate coverage and working staged PDF hyperlinks.
- **Positive**: Missing manifests or source read failures deterministically enter `TraceabilityScanState::Failed` with `Retry` recovery rather than misreporting orphaned manuscripts.
- **Neutral**: Compressed WebAssembly bundles remain unchanged in runtime dependencies and stay well below the 3 MiB `.wasm.br` / `.wasm.gz` budget.
