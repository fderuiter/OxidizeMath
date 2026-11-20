# ADR 0027: Complete Build Rerun Directives and Upgrade Cargo Cache to Node 20

## Status
Accepted (extends [ADR 0002](0002-portable-build-paths.md))

## Context
Restored `target/` build caches in `.github/actions/cargo-cache/action.yml` and local incremental builds exposed three cache-invalidation and runner-runtime gaps:
1. **Zero Rerun Directives in `oxidize_core`**: `crates/oxidize_core/build.rs` walked `../../papers`, `../../math_explorer/src`, and `../../math_explorer_gui/src/tabs` to generate `$OUT_DIR/vfs_data.rs` for `WasmVfs::list_dir`, while emitting zero `cargo:rerun-if-changed` lines. Cargo's default package-root scan ignored external workspace directories, leaving cached `vfs_data.rs` tables stale when manuscripts or domain files were added or removed.
2. **Untracked External Path Dependencies in `math_explorer_gui`**: `math_explorer_gui/build.rs` emitted `cargo:rerun-if-changed` only for `src/tabs` and `Cargo.toml`, while `scan_cargo_toml_deps` and `scan_external_crate` traversed six workspace path dependencies (`federated_registry`, `oxidize_core`, `math_explorer`, `math_commons`, `scientific_metadata`, and `diagnostics`) using `env::current_dir()` without emitting `cargo:rerun-if-changed` lines for external `src/` directories or `.rs` files.
3. **Deprecated `actions/cache` v3 Runtime and Unhashed `Cargo.toml` Manifests**: `.github/actions/cargo-cache/action.yml` pinned `actions/cache@6f8efc29b200d32929f49075959781ed54ec270c` (`v3.5.0`, running on deprecated `node16`) and omitted `'Cargo.toml'` and `'**/Cargo.toml'` from `LOCK_AND_CONFIG_HASH`, causing manifest-only lint, feature, or release-profile edits to reuse unchanged primary cache keys.

## Decision
1. **Emit Normalized Rerun Directives in `crates/oxidize_core/build.rs`**: Emit `cargo:rerun-if-changed=build.rs`, `cargo:rerun-if-changed=src/path_utils.rs`, and forward-slash normalized `cargo:rerun-if-changed` lines via `path_utils::normalize_path` for every walked directory in `../../papers`, `../../math_explorer/src`, and `../../math_explorer_gui/src/tabs` using deterministic sorted iterative traversal.
2. **Emit Portable Rerun Directives in `math_explorer_gui/build.rs`**: Import `path_utils` in `math_explorer_gui/build.rs`, resolve path dependencies via manifest-relative paths (`path_utils::join_and_normalize(dep_path, "src")`) consistent with [ADR 0002](0002-portable-build-paths.md), walk dependency `src/` trees iteratively without recursion, and emit normalized `cargo:rerun-if-changed` lines for `build.rs`, local tab modules, and every walked dependency `src/` directory and `.rs` source file.
3. **Upgrade `cargo-cache` to `actions/cache` v4.3.0 and Hash Workspace Manifests**: Pin both cache steps in `.github/actions/cargo-cache/action.yml` to `actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830` (`v4.3.0` on `node20`), include `'Cargo.toml'` and `'**/Cargo.toml'` inside `LOCK_AND_CONFIG_HASH`, and keep `~/.cargo/bin` excluded from cached paths.
4. **Automated `OUT_DIR` and Workflow Regression Tests**: Add unit tests in `oxidize_core`, `math_explorer_gui`, and `unified_verification` inspecting Cargo `OUT_DIR` parent `output` files and `.github/actions/cargo-cache/action.yml` invariants.

## Consequences
- **Positive**: Warm `target/` builds in CI and local development reliably regenerate `$OUT_DIR/vfs_data.rs` and `$OUT_DIR/generated_tabs.rs` whenever manuscripts, domain modules, or external `ExplorerTab` plugins change.
- **Positive**: All five Rust CI jobs execute `actions/cache` on `node20` pinned to a 40-character commit SHA and rotate primary `target/` cache keys whenever any workspace `Cargo.toml` manifest changes.
- **Neutral**: All public Rust signatures in `crates/oxidize_core/src/` and `crates/unified_verification/api_baseline.json` remain unchanged.
