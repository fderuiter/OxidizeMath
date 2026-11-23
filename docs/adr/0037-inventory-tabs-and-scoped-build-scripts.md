# 37. Static Inventory Tab Registration and Scoped Build Scripts

Date: 2026-11-22

## Status

Accepted (Refines [ADR 0001](./0001-dynamic-workspace-discovery.md) and supersedes the cross-workspace native build-script watchers in [ADR 0027](./0027-build-rerun-directives-and-cargo-cache.md))

## Context

Three build-time code-generation patterns degraded local edit-compile-test feedback loops and hid GUI source files from workspace tooling:

1. **Inline `include!` Tab Wrappers**: `math_explorer_gui/build.rs` scraped `// @explorer_feature` and `// @explorer_order` comments plus external `Cargo.toml` path dependencies to emit `$OUT_DIR/generated_tabs.rs`, which `math_explorer_gui/src/tabs/mod.rs` loaded via `include!`. Because `rustfmt` does not follow `include!(concat!(env!("OUT_DIR"), ...))`, tab modules under `math_explorer_gui/src/tabs/` were invisible to `cargo fmt --all -- --check`, disallowed inner attributes (`#![...]`), and required manual `#[path = "..."]` workarounds in `traceability.rs` and directory-fallback workarounds in `oxidize_core::traceability`.
2. **Broad Native Build-Script Watchers & Unconditional Writes**:
   - `math_explorer_gui/build_snapshot_scan.rs` watched all 16 `crates/*/src` directories (including host-only verification and macro crates and disabled Cargo features) and `build_snapshot.rs` overwrote `traceability_manifest.json` unconditionally.
   - `crates/oxidize_core/build.rs` walked `papers/`, `math_explorer/src`, and `math_explorer_gui/src/tabs` on native host builds even though `crates/oxidize_core/src/vfs.rs` only embeds `$OUT_DIR/vfs_data.rs` on `wasm32`.
3. **Unsorted and Duplicate Markdown Doctests**: `crates/markdown_tests` used Rust 2021, walked directories in filesystem order, emitted anonymous `md_test_0..N` module names, overwrote `$OUT_DIR/generated_tests.rs` unconditionally, and recompiled library crate `README.md` doctests already verified via `#![doc = include_str!("../README.md")]`.

## Decision

1. **Explicit Feature-Gated Tab Modules & `inventory` Collection**:
   - Replace `$OUT_DIR/generated_tabs.rs` and substring scraping in `math_explorer_gui/build.rs` with explicit `#[cfg(feature = "...")] pub mod <tab>;` declarations in `math_explorer_gui/src/tabs/mod.rs` and static `TabDescriptor` registration via `inventory::collect!(TabDescriptor)` / `inventory::submit!`, sorted deterministically by `(order, id)` on both desktop and `wasm32`.
   - Upgrade `oxidize_core::traceability` and `math_explorer_gui/build_snapshot_scan.rs` to traverse `mod.rs` entrypoints and `#[path = "..."]` submodule attributes directly via standard Rust module resolution, removing `#[path]` workarounds from `math_explorer_gui/src/tabs/traceability.rs` and directory-fallback walks from `crates/oxidize_core/src/traceability/runner.rs`.
2. **Feature- and Target-Scoped Watchers with Content-Checked Writes**:
   - Scope `math_explorer_gui/build_snapshot_scan.rs` to active runtime and `CARGO_FEATURE_*`-enabled dependency crates plus enabled GUI tab modules, preserving full 28-paper traceability parity under `--all-features` and default features while writing `traceability_manifest.json` only when bytes change.
   - Gate VFS directory walks in `crates/oxidize_core/build.rs` on `CARGO_CFG_TARGET_ARCH == "wasm32"`, emitting sorted VFS tables on `wasm32` and a minimal stub on native targets via content-checked writes without watching `papers/`, `math_explorer/src`, or `math_explorer_gui/src/tabs` on host builds.
3. **Rust 2024 Deterministic Path-Named Markdown Doctests**:
   - Upgrade `crates/markdown_tests/Cargo.toml` to `edition = "2024"`, sort directory and file paths deterministically, skip library crate `README.md` files already covered by `#![doc = include_str!("../README.md")]`, emit path-derived module identifiers, and content-check `$OUT_DIR/generated_tests.rs` writes.

## Consequences

- **Positive**: `cargo fmt --all -- --check` automatically formats and verifies every `.rs` file under `math_explorer_gui/src/tabs/`.
- **Positive**: Editing manuscripts or unrelated crates no longer triggers native `oxidize_core` rebuilds or rewrites unchanged `$OUT_DIR` artifacts, preserving Cargo modification-time caches.
- **Positive**: `cargo test --doc --workspace --all-features` compiles each crate `README.md` code fence once and reports clear path-derived module names when repository Markdown examples fail.
- **Neutral**: Adding a new GUI domain tab requires declaring `pub mod <tab>;` in `math_explorer_gui/src/tabs/mod.rs` and submitting a `TabDescriptor` via `inventory::submit!`, matching `StarterPresetMetadata` and `ToolMetadata`.
