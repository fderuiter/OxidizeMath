# 16. Purge Redundant Dead Code Suppressions and Update Build Script Visibility

Date: 2026-11-15

## Status

Accepted

## Context

Several active library functions and GUI methods retained obsolete `#[allow(dead_code)]` or `#![allow(dead_code)]` annotations and corresponding entries in `verification_whitelist.toml`. Specifically, `crates/oxidize_core/src/path_utils.rs` functions (`normalize_path`, `strip_and_normalize`, `join_and_normalize`) and GUI helpers (`can_undo`, `can_redo`, `tab_names`, `render_main_menu`, and geometry topology export utils) are in active use across the codebase.

Furthermore, `crates/oxidize_core/build.rs` and `crates/markdown_tests/build.rs` include `path_utils.rs` via path attributes during build execution. Without dead code suppressions in `path_utils.rs`, the build scripts generated compiler dead code warnings because individual build scripts only invoke a subset of path utility functions.

## Decision

1. Remove `#[allow(dead_code)]` suppressions from active functions in `crates/oxidize_core/src/path_utils.rs`, `math_explorer_gui/src/app/session.rs`, `math_explorer_gui/src/app/mod.rs`, `math_explorer_gui/src/app/ui_panels.rs`, and module attribute from `math_explorer_gui/src/tabs/geometry_topology/export_utils.rs`.
2. Purge stale whitelist records from `verification_whitelist.toml`.
3. Expose `pub mod path_utils;` in `crates/oxidize_core/build.rs` and `crates/markdown_tests/build.rs` with doc comments so that the module is treated as a public module root within build scripts, preventing compiler dead code warnings on build scripts without requiring bypass attributes.

## Consequences

- Eliminates redundant dead code suppressions and keeps `verification_whitelist.toml` strictly minimal.
- Ensures workspace builds cleanly with zero dead code compiler warnings under all profiles.
- Retains policy compliance and architectural verification across workspace crates.
